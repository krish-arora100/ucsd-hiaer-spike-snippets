import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-out-dir",
    default="/home/k7arora/output/misc",
    type=str,
    help="Directory containing accuracy npy files and where plots will be saved",
)
args = parser.parse_args()
output_dir = args.out_dir

# EDIT THESE
bar_plot_title = 'Accuracy vs Model Config'
# Labels below each group (one per group)
# group_labels = [
#     'Ch=1, Conv=3',
#     'Ch=20, Conv=1',
#     'Ch=20, Conv=3',
#     'Ch=20, Conv=5'
# ]
# Number of bars per group determined by the following
# bar_labels = ['Pre-Quantization', 'Post-Quantization', 'Post-Conversion']
# bar_colors = ['blue', 'red', 'green']

# # Example: 4 groups, 3 bars per group
# accuracies = [
#     [0.85, 0.80, 0.75],  # Ch=1, Conv=3
#     [0.88, 0.82, 0.78],  # Ch=20, Conv=1
#     [0.90, 0.85, 0.80],  # Ch=20, Conv=3
#     [0.92, 0.88, 0.84]   # Ch=20, Conv=5
# ]

group_labels = [
    'Ch=1, Conv=3',
    'Ch=20, Conv=1',
    'Ch=20, Conv=3'
]

bar_labels = ['Post-Quantization', 'Post-Conversion']
bar_colors = ['blue', 'red']

# Example: 3 groups, 2 bars per group
accuracies = [
    [0.85, 0.45],  # Ch=1, Conv=3
    [0.88, 0.5],  # Ch=20, Conv=1
    [0.90, 0.34],  # Ch=20, Conv=3
]

# LOAD ACCURACIES FROM ALL .NPY FILES IN OUTPUT DIR if available
# npy_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
# accuracies = []
# for fname in sorted(npy_files):
#     npy_path = os.path.join(output_dir, fname)
#     try:
#         arr = np.load(npy_path)
#         accuracies.append(arr.tolist())
#     except Exception as e:
#         print(f"Warning: Could not load {npy_path}: {e}")



# CREATE BAR PLOT

num_groups = len(group_labels)
num_bars_per_group = len(bar_labels)

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.10 
intra_bar_gap = bar_width
group_gap = bar_width * 4
# Calculate group width
group_width = bar_width * num_bars_per_group + intra_bar_gap * (num_bars_per_group - 1) + group_gap
indices = np.arange(num_groups) * group_width

for i in range(num_bars_per_group):
    values = [acc[i] for acc in accuracies]
    bars = ax.bar(indices + i * (bar_width + intra_bar_gap), values, width=bar_width, color=bar_colors[i], label=bar_labels[i])
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)


# thin vertical lines exactly centered between groups
for i in range(1, len(indices)):
    last_bar_x = indices[i-1] + (num_bars_per_group - 1) * (bar_width + intra_bar_gap)
    first_bar_x = indices[i]
    divider_x = (last_bar_x + first_bar_x) / 2
    ax.axvline(x=divider_x, color='lightgrey', linestyle='-', linewidth=1, zorder=0)

ax.set_xticks(indices + ((num_bars_per_group - 1) / 2) * (bar_width + intra_bar_gap))
ax.set_xticklabels(group_labels, fontsize=12)
ax.set_xlabel('Model Config', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)


# legend and title
ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=num_bars_per_group)
ax.set_title(bar_plot_title, fontsize=15, y=1.10)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# save plot to output_dir/plots
plot_dir = os.path.join(output_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)
save_path = os.path.join(plot_dir, "accuracy_barplot.png")
plt.savefig(save_path)
print(f"Saved: {save_path}")