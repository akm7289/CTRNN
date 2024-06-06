import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 3))

# Create input, neuron, and output boxes
input_box = patches.Rectangle((0.2, 0.6), 0.3, 0.2, linewidth=2, edgecolor='blue', facecolor='lightblue', label='Input')
neuron_box = patches.Rectangle((0.2, 0.3), 0.3, 0.2, linewidth=2, edgecolor='green', facecolor='lightgreen', label='Neuron')
output_box = patches.Rectangle((0.2, 0.0), 0.3, 0.2, linewidth=2, edgecolor='red', facecolor='lightcoral', label='Output')

# Add boxes to the plot
ax.add_patch(input_box)
ax.add_patch(neuron_box)
ax.add_patch(output_box)

# Add labels for tau and arrows
ax.annotate('τ', xy=(0.25, 0.24), fontsize=16)
ax.annotate('↑', xy=(0.5, 0.43), fontsize=12)
ax.annotate('↓', xy=(0.5, 0.18), fontsize=12)

# Set axis limits and labels
ax.set_xlim(0, 0.7)
ax.set_ylim(0, 0.8)
ax.set_aspect('equal', adjustable='box')
ax.set_xticks([])
ax.set_yticks([])

# Add legend
ax.legend()

# Add a title
plt.title('CTRNN Cell with Input, Output, and τ')

# Show the plot
plt.show()
