import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load CSV
df = pd.read_csv("output.csv")

# Extract method and parameter
df['method'] = df['combination'].apply(lambda x: x.split()[0])
df['param'] = df['combination'].apply(lambda x: float(x.split()[1]))

# Metrics to plot
metrics = [
    # 'speed',
    'cm_total: mean', #'cm_total: std',
    'rep_total: mean', #'rep_total: std',
    'total features',
    'num possible correct matches',
    'Matching distance mAP','Matching average_response mAP','Matching average_ratio mAP',
    'Verification distance mAP','Verification average_response mAP','Verification average_ratio mAP',
    'Retrieval distance mAP','Retrieval average_response mAP','Retrieval average_ratio mAP'
]

# Set up subplots
n_cols = 3
n_rows = (len(metrics) + n_cols - 1) // n_cols  # ceil division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
axes = axes.flatten()

colors = sns.color_palette("tab10")  # Up to 10 distinct colors

for i, metric in enumerate(metrics):
    ax = axes[i]
    methods = df['method'].unique()
    ax_list = [ax]

    # Create extra y-axes if more than 1 method
    for _ in range(len(methods) - 1):
        ax_list.append(ax_list[-1].twinx())

    # Offset extra y-axes
    for j, extra_ax in enumerate(ax_list[1:], start=1):
        extra_ax.spines['right'].set_position(('axes', 1 + 0.1 * j))

    # Map actual x-values to evenly spaced positions
    param_values = np.sort(df['param'].unique())
    x_map = {val: idx for idx, val in enumerate(param_values)}

    # Plot each method
    for j, (method, a) in enumerate(zip(methods, ax_list)):
        data = df[df['method'] == method].copy()
        data['x_pos'] = data['param'].map(x_map)
        sns.lineplot(
            data=data,
            x='x_pos',
            y=metric,
            marker='o',
            ax=a,
            color=colors[j % len(colors)],
            label=method
        )
        a.set_ylabel(method)

    ax.set_title(metric)
    ax.set_xlabel('param')
    ax.set_xticks(list(x_map.values()))
    ax.set_xticklabels(list(x_map.keys()))

    # Combine legends
    lines, labels = [], []
    for a in ax_list:
        l, lab = a.get_legend_handles_labels()
        lines += l
        labels += lab
    ax.legend(lines, labels, loc='best')

# Hide unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()