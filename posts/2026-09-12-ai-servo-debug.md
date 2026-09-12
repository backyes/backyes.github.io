---
title: "用 AI 取代手工调试：串口舵机一键校准实战"
date: 2026-09-12
tags: ["AI", "Hardware", "Servo", "Serial", "Python", "HLS", "FEETech"]
excerpt: "从手动扫端口、试波特率、改 ID、做中位校准，到 AI 一句话全搞定。用 Hermes Agent 驱动整个串口舵机调试流程，5 分钟完成传统 30-60 分钟的手工活。"
---

# 用 AI 取代手工调试：串口舵机一键校准实战

> 从手动扫端口、试波特率、改 ID、做中位校准，到 AI 一句话全搞定。

## 背景

最近调试飞腾（FEETech）HLS 系列串口舵机（型号 0903），需要完成：
1. 找到串口设备
2. 确定波特率
3. 修改舵机 ID
4. 中位校准

传统做法：打开串口助手 → 手动试波特率 → 发十六进制指令 → 逐字节核对校验和 → 改 ID → 校准……每一步都要查手册、算校验、试错。

这次我直接用 AI（Hermes Agent）驱动整个流程，从扫描到校准 5 分钟完成。下面是完整实战记录。

---

## 第一步：找设备

插入 USB 转串口芯片（CH343P）后，先列出所有串口：

```bash
ls /dev/tty.* /dev/cu.*
```

AI 一眼识别出 `cu.usbmodem5B790502931` 是目标设备（`usbmodem` 前缀 = USB 转串口），其他是蓝牙和调试口直接过滤掉。

**传统做法：** 打开设备管理器，一个个看描述，拔插对比哪个是新出现的。

---

## 第二步：确定波特率

SC 舵机常用波特率从 9600 到 1Mbps 共十几种。AI 写了一个自动扫描脚本：

```python
import serial
import time

port = '/dev/cu.usbmodem5B790502931'

for baud in [1000000, 500000, 250000, 115200, 9600]:
    ser = serial.Serial(port=port, baudrate=baud, timeout=0.3)
    data = [0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB]
    ser.write(bytes(data))
    time.sleep(0.1)
    if ser.in_waiting:
        resp = ser.read(ser.in_waiting)
        print(f"@{baud}: {resp.hex()}")
    ser.close()
```

输出：
```
@1000000: b'\xff\xff\x01\x02\x00\xfc'
```

**1Mbps 命中。** AI 还解析了响应帧：`FF FF` = 帧头，`01` = 舵机 ID，`02` = 长度，`00` = 无错误，`FC` = 校验和（通过）。

---

## 第三步：读取状态

确认通信后，AI 读取了舵机关键寄存器：

| 寄存器 | 地址 | 值 | 含义 |
|--------|------|-----|------|
| 型号 | 0x03 | 0x0903 | HLS 系列 |
| 当前位置 | 0x38 | 2640 | 约 773° |
| 目标位置 | 0x2A | 1 | 约 0.3° |
| 扭矩开关 | 0x28 | 0 | 关闭 |

---

## 第四步：修改 ID（第二个舵机）

串接第二个舵机后，两个默认都是 ID=1，会冲突。AI 执行修改流程：

```python
# 1. 解锁 EPROM（Lock 寄存器 = 55）
write_byte(ser, 1, 55, 0)
# 2. 改 ID（ID 寄存器 = 5）
write_byte(ser, 1, 5, 2)
# 3. 锁定
write_byte(ser, 2, 55, 1)
```

验证：`Ping ID=2` → 在线 ✅

---

## 第五步：中位校准

HLS 系列的中位校准是向扭矩开关寄存器写特殊值 `128`：

```python
# 中位校准（地址 40 = Torque Enable）
write_byte(ser, 2, 40, 128)
# 开启扭矩
write_byte(ser, 2, 40, 1)
```

校准后位置：2 → **2048**（中位，即 180° 位置）。

---

## AI 调试 vs 手工调试对比

| 环节 | 手工 | AI 驱动 |
|------|------|---------|
| 找端口 | 看设备管理器 | 一眼识别 |
| 试波特率 | 串口助手逐个试 | 脚本自动扫 |
| 算校验和 | 手算 | 自动计算 |
| 改 ID | 发 3 条指令+手动验证 | 一键完成+自动验证 |
| 整体时间 | **30-60 分钟** | **5 分钟** |

---

## 核心代码

```python
class HLSServo:
    def __init__(self, port, baud=1000000):
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.3)
    
    def _write(self, id, inst, params):
        data = [id, len(params) + 2, inst] + params
        chk = (~sum(data)) & 0xFF
        self.ser.write(bytes([0xFF, 0xFF] + data + [chk]))
        time.sleep(0.05)
    
    def ping(self, id):
        self._write(id, 0x01, [])
        time.sleep(0.03)
        resp = self.ser.read(self.ser.in_waiting)
        return len(resp) >= 6 and resp[0] == 0xFF
    
    def write_byte(self, id, addr, val):
        self._write(id, 0x03, [addr, val])
    
    def set_id(self, old_id, new_id):
        self.write_byte(old_id, 55, 0)   # Unlock
        self.write_byte(old_id, 5, new_id)
        self.write_byte(new_id, 55, 1)   # Lock
        return self.ping(new_id)
    
    def calibrate(self, id):
        pos_before = self.read_word(id, 56)
        self.write_byte(id, 40, 128)     # 校准
        time.sleep(0.3)
        self.write_byte(id, 40, 1)       # 开扭矩
        return pos_before, self.read_word(id, 56)
```

---

## 总结

1. **AI 最擅长"查表+执行"** — 从技术文档找寄存器地址，自动构造指令帧
2. **校验和计算交给 AI** — 人容易算错，AI 不会
3. **AI 也会犯错** — 我第一次校准逻辑用的第三方 Wiki 而非官方文档，所以一定要给 AI 正确的资料来源

**一句话：让 AI 做"查手册+算校验+发指令"的苦力活，你只做决策。**
