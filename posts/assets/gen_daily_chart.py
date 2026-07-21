#!/usr/bin/env python3
"""Generate daily cost comparison chart: DeepSeek vs Kimi (storage + compute breakdown)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict

# Parse CSV
data = defaultdict(lambda: {'cache_hit': 0, 'cache_miss': 0, 'output': 0})
with open('amount-2026-07.csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        d = row['utc_date']; t = row['type']; a = int(row['amount'])
        if 'cache_hit' in t: data[d]['cache_hit'] = a
        elif 'cache_miss' in t: data[d]['cache_miss'] = a
        elif 'output' in t: data[d]['output'] = a

dates = sorted(data.keys())
cache_hit = [data[d]['cache_hit']/1e6 for d in dates]
miss = [data[d]['cache_miss']/1e6 for d in dates]
out = [data[d]['output']/1e6 for d in dates]

# Prices
DS_HIT, DS_MISS, DS_OUT = 0.003625, 0.435, 0.87
KI_HIT, KI_MISS, KI_OUT = 0.28, 2.78, 13.90

ds_storage = [h*DS_HIT for h in cache_hit]
ds_compute = [m*DS_MISS + o*DS_OUT for m,o in zip(miss, out)]
ki_storage = [h*KI_HIT for h in cache_hit]
ki_compute = [m*KI_MISS + o*KI_OUT for m,o in zip(miss, out)]

x = np.arange(len(dates))
w = 0.2

fig, ax = plt.subplots(figsize=(14, 7))
b1 = ax.bar(x-1.5*w, ds_storage, w, label='DeepSeek Storage', color='#4a8fe0', alpha=0.85)
b2 = ax.bar(x-0.5*w, ds_compute, w, label='DeepSeek Compute', color='#7da7e8', alpha=0.85)
b3 = ax.bar(x+0.5*w, ki_storage, w, label='Kimi Storage', color='#d29922', alpha=0.85)
b4 = ax.bar(x+1.5*w, ki_compute, w, label='Kimi Compute', color='#e8c547', alpha=0.85)

# Line for ratio
ax2 = ax.twinx()
ratio = [ki_storage[i]+ki_compute[i]/(ds_storage[i]+ds_compute[i]) if (ds_storage[i]+ds_compute[i])>0 else 0 for i in range(len(dates))]
ratio = [(ki_storage[i]+ki_compute[i])/(ds_storage[i]+ds_compute[i]) for i in range(len(dates))]
ax2.plot(x, ratio, 'r^-', linewidth=2.5, markersize=10, label='Kimi/DeepSeek Ratio', color='#e04040')
ax2.set_ylabel('Cost Ratio (Kimi / DeepSeek)', fontsize=12, color='#e04040')
ax2.tick_params(axis='y', labelcolor='#e04040')
ax2.set_ylim(0, max(ratio)*1.3)
for i,r in enumerate(ratio):
    ax2.annotate(f'{r:.1f}×', xy=(i,r), xytext=(0,10), textcoords='offset points', ha='center', fontsize=10, color='#e04040', fontweight='bold')

ax.set_ylabel('Daily Cost ($)', fontsize=12)
ax.set_title('Daily Cost Breakdown: DeepSeek Pro (V4) vs Kimi K3', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([d[5:] for d in dates], fontsize=11)
ax.legend(loc='upper left', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(ki_storage+ki_compute)*1.25)

# Value labels
for b in b1:
    h=b.get_height()
    if h>0.1: ax.annotate(f'${h:.2f}', xy=(b.get_x()+b.get_width()/2,h), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8, color='#4a8fe0')
for b in b3:
    h=b.get_height()
    if h>0.5: ax.annotate(f'${h:.0f}', xy=(b.get_x()+b.get_width()/2,h), xytext=(0,3), textcoords='offset points', ha='center', fontsize=8, color='#d29922')

plt.tight_layout()
fig.text(0.99, 0.01, 'backyes.github.io', fontsize=8, color='#8b949e',
         ha='right', va='bottom', alpha=0.6, style='italic')
plt.savefig('daily_cost_comparison.png', dpi=150, bbox_inches='tight')
print("Chart saved: daily_cost_comparison.png")
