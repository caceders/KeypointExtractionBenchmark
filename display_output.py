import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("output.csv")

# Extract method and parameter
df['method'] = df['combination'].apply(lambda x: x.split()[0])
df['param'] = df['combination'].apply(lambda x: float(x.split()[1]))

# Metrics to plot
metrics = [
    'speed',
    'cm_total: mean', 'cm_total: std',
    'rep_total: mean', 'rep_total: std',
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

for i, metric in enumerate(metrics):
    sns.lineplot(data=df, x='param', y=metric, hue='method', marker='o', ax=axes[i])
    axes[i].set_title(metric)
    axes[i].legend(loc='best')

# Hide unused subplots if any
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()