---
title: "让AI帮我调舵机，结果真香了 | 串口舵机一键校准实战"
date: 2026-09-12
tags: ["AI", "硬件", "舵机", "串口", "Python", "FEETech", "B站"]
excerpt: "用AI助手（Hermes Agent）全程驱动串口舵机调试：从扫描端口→确定波特率→改ID→中位校准，5分钟搞定传统30-60分钟的手工活。附视频演示+完整代码。"
---

# 让 AI 帮我调舵机，结果真香了

> 一句话总结：**把"查手册+算校验+发指令"的苦力活丢给 AI，你只做决策。**

## 视频导航

> 完整演示视频见 B 站同名文章上方 👆

| 时间 | 内容 | 本文对应章节 |
|------|------|-------------|
| 00:00 | 开场 & 背景介绍 | → [背景](#背景) |
| 01:23 | 找设备 + 扫波特率 | → [第一二步](#第一步找端口) |
| 02:45 | 读状态 + 看寄存器 | → [第三步读状态](#第三步读状态) |
| 04:10 | 串接第二台 + 改 ID | → [第四步改-id](#第四步修改-id) |
| 06:30 | 中位校准演示 | → [第五步校准](#第五步中位校准) |
| 08:00 | 总结 & 对比 | → [对比总结](#ai-vs-手工对比) |

---

## 背景

最近在调飞腾（FEETech）HLS 系列串口舵机，型号 0903，USB 转串口芯片是 CH343P。

**需要干的事：**

- [x] 找串口设备
- [x] 确定波特率
- [x] 串接第二台、改 ID 避免冲突
- [x] 中位校准
- [x] 写个一键脚本

**传统路径：** 打开串口助手 → 手动试波特率 → 查手册找寄存器地址 → 算十六进制校验和 → 发指令 → 看回复 → 解析 → 错了重来……

**本次路径：** 告诉 AI 我要干嘛，5 分钟全搞定。

---

## 第一步：找端口

```bash
$ ls /dev/cu.*
```

AI 一眼锁定 `cu.usbmodem5B790502931`：

> `usbmodem` 前缀 = USB 转串口设备，其他 `Bluetooth-Incoming-Port`、`debug-console` 直接过滤。

**省掉的操作：** 不用打开设备管理器、不用拔插对比、不用看描述猜哪个是目标。

---

## 第二步：扫波特率

串口舵机常用波特率从 9600 到 1Mbps 有十几种可能。AI 写了个自动扫描：

```python
for baud in [1000000, 500000, 250000, 115200, 9600]:
    ser = serial.Serial(port=port, baudrate=baud, timeout=0.3)
    ser.write(bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB]))  # Ping
    time.sleep(0.1)
    if ser.in_waiting:
        print(f"@{baud}: {ser.read(ser.in_waiting).hex()}")
```

结果秒出：

```
@1000000: b'\xff\xff\x01\x02\x00\xfc'
```

**1Mbps 命中。** AI 顺便解析了响应帧：

| 字节 | 含义 |
|------|------|
| `FF FF` | 帧头 |
| `01` | 舵机 ID |
| `02` | 数据长度 |
| `00` | 错误码（0=正常）|
| `FC` | 校验和（计算通过 ✓）|

---

## 第三步：读状态

通信确认后，AI 一口气读出关键寄存器：

| 功能 | 地址 | 值 | 说明 |
|------|------|-----|------|
| 型号 | 0x03 | `0903` | HLS 系列 |
| 当前位置 | 0x38 | 2640 | ≈773° |
| 目标位置 | 0x2A | 1 | ≈0.3° |
| 扭矩开关 | 0x28 | 0 | 关闭 |
| 电压 | 0x3E | 0 | 不支持检测 |
| 温度 | 0x3F | 0 | 不支持检测 |

> ⚡ **避坑提示：** 电压/温度返回 0 不代表舵机坏了，是该型号根本不支持这两个寄存器。

---

## 第四步：修改 ID

串接第二台舵机后，**两个默认 ID 都是 1，会冲突！** 需要把新的改为 ID=2。

**SC 协议改 ID 三步走：**

```python
# 1. 解锁 EPROM（Lock 寄存器地址=55）
write_byte(ser, old_id=1, addr=55, val=0)

# 2. 写入新 ID（ID 寄存器地址=5）
write_byte(ser, old_id=1, addr=5, val=2)

# 3. 锁定 EPROM
write_byte(ser, new_id=2, addr=55, val=1)
```

**⚠️ 关键：** 第三步锁定必须用**新 ID（2）** 来发，否则新 ID 不会生效。

验证：

```
Ping ID=1: 在线
Ping ID=2: 在线 ← 新舵机已识别
```

> 💡 **提示：** 修改 ID 后建议**断电重启**一次，让新 ID 完全写入非易失存储。

---

## 第五步：中位校准

**这是最容易踩坑的一步。**

很多人（包括我第一次）会想："中位校准不就是写个偏移量到偏移寄存器吗？"

错！HLS 系列的校准方式是：

```
向"扭矩开关"寄存器（地址 40）写一个特殊值 128
```

舵机收到 128 后，自动把**当前位置**记录为零点偏移。

```python
# 中位校准
write_byte(ser, id=2, addr=40, val=128)
time.sleep(0.3)   # 等舵机响应

# 开启扭矩
write_byte(ser, id=2, addr=40, val=1)
```

**校准效果：**

| 校准前 | 校准后 |
|--------|--------|
| 位置：2 | 位置：**2048**（= 180° 中位） |
| 偏移：0 | 偏移：4093（自动记录） |

> 🎯 写 128 校准 → 写 1 开扭矩，这是 HLS/SMS_STS 系列的"潜规则"，手册里不会明显标注。

---

## AI vs 手工对比

| 环节 | 手工 | AI |
|------|------|-----|
| 找端口 | 设备管理器翻找 | 一条命令识别 |
| 试波特率 | 串口助手逐个试，每个等 1-2 秒 | 脚本自动扫，1 秒命中 |
| 算校验和 | 手算 `~sum & 0xFF`，容易出错 | 自动生成+自动验证 |
| 查寄存器 | 翻 PDF 手册逐页找 | 直接从 Wiki 获取 |
| 改 ID | 3 条指令 + 手动验证 | 一键完成 |
| 中位校准 | 不知道写 128 到地址 40 | 直接知道"潜规则" |
| **总耗时** | **30-60 分钟** | **≈5 分钟** |

---

## 完整代码

```python
#!/usr/bin/env python3
"""串口舵机一键扫描 + 中位校准 | HLS/SMS_STS 系列兼容"""

import serial
import time


class HLSServo:
    """飞腾 HLS / SMS_STS 系列串口舵机控制类"""

    def __init__(self, port: str, baud: int = 1000000):
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.3)

    def _write(self, id: int, inst: int, params: list):
        """构造指令帧并发送"""
        data = [id, len(params) + 2, inst] + params
        chk = (~sum(data)) & 0xFF
        self.ser.write(bytes([0xFF, 0xFF] + data + [chk]))
        time.sleep(0.05)

    def ping(self, id: int) -> bool:
        """Ping 舵机，返回是否在线"""
        self._write(id, 0x01, [])
        time.sleep(0.03)
        resp = self.ser.read(self.ser.in_waiting)
        return len(resp) >= 6 and resp[0] == 0xFF and resp[1] == 0xFF

    def read_word(self, id: int, addr: int) -> int:
        """读 16 位寄存器"""
        self._write(id, 0x02, [addr, 2])
        resp = self.ser.read(self.ser.in_waiting)
        if len(resp) >= 7:
            return resp[5] + (resp[6] << 8)
        return -1

    def write_byte(self, id: int, addr: int, val: int):
        """写单字节到寄存器"""
        self._write(id, 0x03, [addr, val])

    def set_id(self, old_id: int, new_id: int) -> bool:
        """修改舵机 ID"""
        self.write_byte(old_id, 55, 0)       # 解锁
        self.write_byte(old_id, 5, new_id)   # 改 ID
        self.write_byte(new_id, 55, 1)       # 锁定（用新 ID）
        time.sleep(0.2)
        return self.ping(new_id)

    def calibrate(self, id: int) -> tuple:
        """中位校准，返回 (校准前位置, 校准后位置)"""
        before = self.read_word(id, 56)
        self.write_byte(id, 40, 128)         # 校准指令
        time.sleep(0.3)
        self.write_byte(id, 40, 1)           # 开扭矩
        after = self.read_word(id, 56)
        return before, after

    def scan(self, max_id: int = 253) -> list:
        """扫描总线上的所有舵机"""
        return [i for i in range(max_id + 1) if self.ping(i)]

    def close(self):
        self.ser.close()


if __name__ == '__main__':
    servo = HLSServo('/dev/cu.usbmodem5B790502931')

    # 1. 扫描
    print("🔍 扫描中...")
    found = servo.scan()
    print(f"✅ 找到舵机: {found}")

    # 2. 逐个校准
    for sid in found:
        before, after = servo.calibrate(sid)
        print(f"ID={sid}: 位置 {before} → {after}")

    servo.close()
    print("🎉 全部完成")
```

---

## 踩坑总结

| 坑 | 原因 | 解决方案 |
|----|------|----------|
| 校准写偏移寄存器没用 | HLS 系列不是写偏移量，是写 128 到扭矩寄存器 | `write_byte(id, 40, 128)` |
| 改 ID 后不生效 | 锁定必须用新 ID 发 | 第三步用 `new_id` 发 Lock |
| 两个 ID=1 冲突 | 出厂默认都是 ID=1 | 先改第二个舵机的 ID |
| 电压/温度返回 0 | 该型号不支持这两个寄存器 | 忽略即可 |

---

## 相关资源

- 📺 **B 站视频：** 本文对应的手把手演示视频见上方
- 💻 **代码仓库：** [FT_SCServo_Debug_Qt](https://github.com/Kotakku/FT_SCServo_Debug_Qt)
- 📖 **参考文档：** [AIFITLAB HLS 内存表](https://wiki.aifitlab.com/feetech-servo-motor-docs/feetech-hls-servo-memory-table-analysis)

---

**如果对你有帮助，欢迎一键三连 👍，视频区见！**

> 💬 评论区提问：你调试串口设备时踩过什么坑？
