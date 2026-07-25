#!/usr/bin/env python3
"""Generate hand-drawn style product comparison diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.xkcd()
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('#faf8f5')
ax.set_facecolor('#faf8f5')

# 标题
ax.text(7, 8.5, 'Server-Side CPU: Product Comparison', fontsize=18, fontweight='700',
        ha='center', va='center', color='#2d3436',
        fontfamily='sans-serif')

# 产品数据
products = [
    {'name': 'Volcano Agent\nSandbox', 'x': 1.5, 'color': '#e17055',
     'features': ['Multi-tenant\nisolation', 'Container\norchestration', '10K concurrent\nsandboxes'],
     'cpu': 'Orchestration\n+ Tool exec'},
    {'name': 'Meituan\nInternal Agent', 'x': 3.8, 'color': '#0984e3',
     'features': ['Internal API\naccess', 'RPC middleware', 'Order/Rider\nsystems'],
     'cpu': 'RPC calls\n+ Auth/Rate limit'},
    {'name': 'Kimi\nDeep Research', 'x': 6.1, 'color': '#00b894',
     'features': ['Parallel\nsearch', 'Headless\nbrowsers', 'Data\nprocessing'],
     'cpu': 'Browsers\n+ Parsing'},
    {'name': 'E2B/Modal\nSandbox', 'x': 8.4, 'color': '#6c5ce7',
     'features': ['Firecracker\nMicroVM', '125ms\nstartup', 'Serverless\nscaling'],
     'cpu': 'MicroVM\norchestration'},
    {'name': 'Manus\nFull-Agent', 'x': 10.7, 'color': '#fdcb6e',
     'features': ['Autonomous\nexecution', 'Browser\ncluster', 'Code\nsandbox'],
     'cpu': 'Browsers\n+ Code exec\n+ State mgmt'},
    {'name': 'DeepSeek\nSearch', 'x': 13.0, 'color': '#e84393',
     'features': ['Parallel\nsearch engines', 'Real-time\ndata', 'Deep\nreasoning'],
     'cpu': 'Search cluster\n+ Data pipeline'},
]

# 绘制每个产品卡片
for p in products:
    x = p['x']
    # 卡片背景
    box = FancyBboxPatch((x-1.1, 3.2), 2.2, 4.5,
                          boxstyle="round,pad=0.1",
                          facecolor='white', edgecolor=p['color'],
                          linewidth=2.5, alpha=0.95)
    ax.add_patch(box)
    
    # 产品名
    ax.text(x, 7.3, p['name'], fontsize=11, fontweight='700',
            ha='center', va='center', color=p['color'])
    
    # 分隔线
    ax.plot([x-0.9, x+0.9], [6.9, 6.9], color=p['color'], linewidth=1, alpha=0.4)
    
    # 特性列表
    for i, feat in enumerate(p['features']):
        ax.text(x, 6.3 - i*0.55, feat, fontsize=8, ha='center', va='center',
                color='#2d3436', fontfamily='sans-serif')
    
    # CPU 标签
    ax.text(x, 4.3, 'CPU:', fontsize=8, fontweight='600', ha='center', color='#636e72')
    ax.text(x, 3.9, p['cpu'], fontsize=7.5, ha='center', va='center',
            color=p['color'], fontweight='600',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffeaa7', alpha=0.6,
                      edgecolor='none'))

# 底部对比表
ax.text(7, 2.6, 'Key Differentiator', fontsize=13, fontweight='700',
        ha='center', color='#2d3436')

comparisons = [
    ('Code Location', 'Client', 'Client', 'Cloud', 'Cloud', 'Cloud', 'Cloud'),
    ('Tool Execution', 'Local', 'Server', 'Server', 'Server', 'Server', 'Server'),
    ('User Interaction', 'Step-by-step', 'Step-by-step', 'Step-by-step', 'API', 'Goal → Result', 'Goal → Result'),
    ('CPU Driver', 'None', 'RPC', 'Browsing', 'Orchestration', 'Autonomous', 'Search'),
]

# 表头
headers = ['', 'Claude Code', 'Volcano', 'Meituan', 'Kimi', 'E2B/Modal', 'Manus', 'DeepSeek']
col_widths = [2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
x_start = 0.3

# 绘制表格
for row_idx, row_data in enumerate(comparisons):
    y = 2.0 - row_idx * 0.45
    x = x_start
    
    for col_idx, cell in enumerate(row_data):
        width = col_widths[col_idx]
        # 交替行背景
        if row_idx % 2 == 0:
            rect = plt.Rectangle((x, y-0.18), width, 0.36, facecolor='#dfe6e9', alpha=0.4)
            ax.add_patch(rect)
        
        # 文字样式
        if col_idx == 0:
            weight = '700'
            color = '#2d3436'
        elif row_idx == 0:
            weight = '600'
            color = '#636e72'
        else:
            weight = '400'
            color = '#2d3436'
        
        ax.text(x + width/2, y, cell, fontsize=7.5, ha='center', va='center',
                fontweight=weight, color=color)
        x += width

# 水印
fig.text(0.99, 0.01, 'backyes.github.io', fontsize=8, color='#b2bec3',
         ha='right', va='bottom', alpha=0.7, style='italic')

plt.tight_layout(pad=0.5)
plt.savefig('posts/assets/product_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#faf8f5')
print("Chart saved: posts/assets/product_comparison.png")
