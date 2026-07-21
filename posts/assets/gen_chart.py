#!/usr/bin/env python3
"""
Generate token usage chart from LongCat-2.0 CSV data.
Input: amount-2026-07.csv (from https://longcat.chat/platform/usage)
Output: token_usage_july_2026.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict

# Parse CSV
data = defaultdict(lambda: {'cache_hit': 0, 'cache_miss': 0, 'output': 0, 'requests': 0})
with open('amount-2026-07.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        date = row['utc_date']
        ttype = row['type']
        amount = int(row['amount'])
        if 'cache_hit' in ttype:
            data[date]['cache_hit'] = amount
        elif 'cache_miss' in ttype:
            data[date]['cache_miss'] = amount
        elif 'output' in ttype:
            data[date]['output'] = amount
        elif 'request' in ttype:
            data[date]['requests'] = amount

dates = sorted(data.keys())
cache_hit_m = [data[d]['cache_hit'] / 1e6 for d in dates]
cache_miss_m = [data[d]['cache_miss'] / 1e6 for d in dates]
output_m = [data[d]['output'] / 1e6 for d in dates]
total_compute_m = [cache_miss_m[i] + output_m[i] for i in range(len(dates))]
requests_list = [data[d]['requests'] for d in dates]

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
x = np.arange(len(dates))
width = 0.35

# Top: Token consumption
bars1 = ax1.bar(x - width/2, cache_hit_m, width, label='Cache Hit (Storage)', color='#4a8fe0', alpha=0.85)
bars2 = ax1.bar(x + width/2, total_compute_m, width, label='Cache Miss + Output (Compute)', color='#d29922', alpha=0.85)

ax1.set_ylabel('Tokens (Millions)', fontsize=12)
ax1.set_title('LongCat-2.0 Daily Token Consumption — July 2026', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels([d[5:] for d in dates], fontsize=11)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(cache_hit_m) * 1.2)

for bar in bars1:
    h = bar.get_height()
    ax1.annotate(f'{h:.1f}M', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, color='#4a8fe0')
for bar in bars2:
    h = bar.get_height()
    ax1.annotate(f'{h:.1f}M', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9, color='#d29922')

# Bottom: Request count
ax2.bar(x, requests_list, width=0.5, color='#3fb950', alpha=0.8)
ax2.set_ylabel('Requests', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels([d[5:] for d in dates], fontsize=11)
ax2.set_title('Daily Request Count', fontsize=12, pad=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for i, v in enumerate(requests_list):
    ax2.annotate(str(v), xy=(i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, color='#3fb950')

plt.tight_layout()
plt.savefig('token_usage_july_2026.png', dpi=150, bbox_inches='tight')
print("Chart saved to token_usage_july_2026.png")
print(f"\nSummary:")
print(f"Total cache hit (storage): {sum(cache_hit_m):.1f}M tokens")
print(f"Total compute (miss+output): {sum(total_compute_m):.1f}M tokens")
print(f"Storage:Compute ratio: {sum(cache_hit_m)/sum(total_compute_m):.1f}:1")
