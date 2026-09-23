import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(5, 8))

# Remove axes
ax.axis("off")

# Text boxes
boxes = [
    ("Training data", 0.5, 0.9),
    ("Choose K", 0.5, 0.75),
    ("Find K nearest\nneighbors", 0.5, 0.55),
    ("Vote among\nneighbors", 0.5, 0.35),
    ("Prediction", 0.5, 0.15),
]

# Draw boxes
for text, x, y in boxes:
    width = 0.35
    height = 0.12
    
    rect = Rectangle(
        (x - width/2, y - height/2),
        width,
        height,
        fill=False,
        linewidth=1.5
    )
    ax.add_patch(rect)
    
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=12
    )

# Draw arrows
arrow_positions = [
    (0.5, 0.84, 0.5, 0.81),
    (0.5, 0.69, 0.5, 0.61),
    (0.5, 0.49, 0.5, 0.41),
    (0.5, 0.29, 0.5, 0.21),
]

for x1, y1, x2, y2 in arrow_positions:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.5)
    )

# Set limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Save image
plt.savefig(
    "knn_flowchart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()