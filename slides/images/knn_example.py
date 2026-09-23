import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

# -----------------------
# 1. Create sample data
# -----------------------
class_A = np.array([
    [1, 5],
    [2, 6],
    [2, 3],
    [3, 4]
])

class_B = np.array([
    [7, 5],
    [8, 6],
    [7, 3],
    [9, 4]
])

# New unknown point
new_point = np.array([[5, 5]])

# Combine data
X = np.vstack([class_A, class_B])

# -----------------------
# 2. Find K nearest neighbors
# -----------------------
K = 5

knn = NearestNeighbors(n_neighbors=K)
knn.fit(X)

distances, indices = knn.kneighbors(new_point)

neighbors = X[indices[0]]

# -----------------------
# 3. Plot
# -----------------------
plt.figure(figsize=(7, 5))

# Plot training samples
plt.scatter(
    class_A[:, 0],
    class_A[:, 1],
    marker="^",
    s=150,
    label="Class A"
)

plt.scatter(
    class_B[:, 0],
    class_B[:, 1],
    marker="o",
    s=150,
    label="Class B"
)

# Plot new point
plt.scatter(
    new_point[:, 0],
    new_point[:, 1],
    marker="*",
    s=300,
    label="New point"
)

# Draw lines to nearest neighbors
for point in neighbors:
    plt.plot(
        [new_point[0, 0], point[0]],
        [new_point[0, 1], point[1]],
        linestyle="--"
    )

# Labels
plt.title(f"KNN Classification (K={K})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.legend()
plt.grid(True)

# Save image
plt.savefig(
    "knn_example.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()