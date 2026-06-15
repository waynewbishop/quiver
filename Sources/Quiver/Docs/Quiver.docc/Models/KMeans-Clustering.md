# K-Means Clustering

Partition data into clusters by iteratively refining centroid positions.

## Overview

K-Means partitions data into `k` clusters by assigning each point to the nearest centroid. This unsupervised algorithm discovers latent structures in unlabeled data by measuring distances between points, making it a primary choice for exploratory analysis.

### How it works

The algorithm iteratively refines centroid positions. It starts by placing `k` centroids at random positions, then alternates between two steps:
1.  Assign each point to the nearest centroid, based on [Euclidean distance](<doc:Linear-Algebra-Primer>).
2.  Update each centroid to the **mean** position of its assigned points.

The cycle continues until centroids stabilize (convergence) or the iteration cap is reached.

![Scatter plot of points grouped into colored clusters with each cluster's centroid marked](diagram-kmeans)

Random initializations produce different outcomes. The `seed` parameter ensures reproducibility, and `bestFit` automates multiple initializations to return the lowest-inertia model.

> Note: A **centroid** is the average position of a cluster's points. If three customers have spending scores `[10, 20, 30]` and incomes `[40, 50, 60]`, their centroid is `[20, 50]`.

### The distance connection

At its core, K-Means relies on Euclidean distance: the straight-line distance between two points in n-dimensional space, computed as √Σ(aᵢ − bᵢ)². Assignment compares squared distances, since the squares preserve the same ordering as the distances themselves and the lowest one wins. The same measure powers nearest-neighbor search in `KNearestNeighbors` and similarity operations in <doc:Similarity-Operations>.

> Note: Distance builds on vector subtraction, where each (aᵢ − bᵢ) term is one element of the difference vector. For a deeper look at how vector arithmetic works geometrically, see [Vectors](https://waynewbishop.github.io/swift-algorithms/20-vectors.html) in Swift Algorithms & Data Structures.

### Fitting a model

`KMeans.fit(data:k:maxIterations:seed:)` runs the iterative algorithm and returns a trained model:

```swift
import Quiver

let data: [[Double]] = [
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
    [8.0, 8.0], [8.5, 7.5], [9.0, 8.5]
]

let model = KMeans.fit(data: data, k: 2, seed: 42)
print(model.labels)    // [0, 0, 0, 1, 1, 1]
```

`bestFit` automates multiple initializations to return the tightest clustering:

```swift
// Run 10 initializations, return the lowest-inertia model
let model = KMeans.bestFit(data: data, k: 2, attempts: 10)
```

### Inspecting results

The model exposes centroids, cluster assignments, and inertia (the sum of squared distances to centroids, where lower is better):

```swift
import Quiver

print(model.labels)      // Point assignments
print(model.centroids)   // Cluster centers
print(model.inertia)     // Total squared distance to centroids
print(model.iterations)  // Convergence iteration count
```

The `clusters(from:)` method groups data points into `Cluster` values:

```swift
import Quiver

let clusters = model.clusters(from: data)
for cluster in clusters {
    print("Center: \(cluster.centroid), points: \(cluster.count)")
}
```

Since `Cluster` conforms to `Sequence`, we can use `for-in` loops or `map` directly on the cluster’s points.

### Predicting new points

`predict(_:)` assigns new points to the nearest centroid without retraining:

```swift
let newPoints: [[Double]] = [[2.0, 2.5], [7.0, 7.0]]
let assignments = model.predict(newPoints)
// [0, 1] — first point near cluster 0, second near cluster 1
```

### Choosing k

Choosing `k` requires balancing complexity and inertia (total squared distance to centroids). The `elbowMethod(data:kRange:maxIterations:seed:)` sweeps `k` and returns `(k, inertia)` pairs. The "elbow" where inertia stops decreasing sharply suggests an appropriate `k`.

```swift
// Sweep k 1 to 5 and compare inertia
let results = KMeans.elbowMethod(data: data, kRange: 1...5, seed: 7)
for result in results {
    print("k=\(result.k): inertia=\(result.inertia)")
}
```

### Full pipeline

Feature scaling is critical, as K-Means distance calculations are sensitive to feature magnitude. `StandardScaler` centers features at zero with unit variance, ensuring all dimensions contribute proportionally.

```swift
// Scale, then cluster
let scaler = StandardScaler.fit(features: data)
let scaled = scaler.transform(data)
let model = KMeans.fit(data: scaled, k: 3, seed: 42)
```

### Organizing with Panel

Using a <doc:Panel> keeps column names attached to data.

```swift
let customers = Panel([
    ("spending", [15.0, 16.0, 17.0, 81.0, 77.0, 76.0, 94.0, 87.0, 90.0]),
    ("income",   [39.0, 81.0, 6.0, 77.0, 40.0, 76.0, 3.0, 72.0, 88.0])
])

let features = customers.toMatrix(columns: ["spending", "income"])
let model = KMeans.fit(data: StandardScaler.fit(features: features).transform(features), k: 3, seed: 42)
```

See <doc:Panel> for the type definition and <doc:Panel-Workflows> for the train-predict workflow with named columns.

### When to use K-Means

K-Means is ideal for customer segmentation, anomaly detection, and exploring structure in unlabeled data. It assumes clusters are spherical and similarly sized.

It struggles with non-spherical shapes, varying cluster densities, and categorical data. Note that K-Means assigns every point to a group, meaning outliers are always forced into the nearest cluster.

### Safe by design

`KMeans` and `Cluster` are immutable Swift structs, ensuring models cannot drift and training data stays separate from the results. `Equatable` conformance allows for simple stability verification between runs.

```swift
let run1 = KMeans.fit(data: points, k: 3, seed: 42)
let run2 = KMeans.fit(data: points, k: 3, seed: 42)
run1 == run2  // true
```

## Topics

### Model
- ``KMeans``
- ``Cluster``

### Training
- ``KMeans/fit(data:k:maxIterations:seed:)``
- ``KMeans/bestFit(data:k:maxIterations:attempts:)``
- ``KMeans/elbowMethod(data:kRange:maxIterations:seed:)``

### Prediction
- ``KMeans/predict(_:)``
- ``KMeans/clusters(from:)``

### Related
- <doc:Machine-Learning-Primer>
- <doc:Nearest-Neighbors-Classification>
- <doc:Feature-Scaling>
- <doc:Pipeline>
