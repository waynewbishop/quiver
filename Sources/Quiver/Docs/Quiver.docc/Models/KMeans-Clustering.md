# K-Means Clustering

Partition data into clusters by iteratively refining centroid positions.

## Overview

K-Means is the most widely used clustering algorithm. Given a set of data points, it groups them into `k` clusters where each point belongs to the cluster with the nearest centroid.

> Important: K-Means is **unsupervised** — it works with data that has no labels. Unlike classifiers like `KNearestNeighbors` and `GaussianNaiveBayes` that learn from labeled examples, K-Means discovers structure on its own by measuring distances between points. This makes it useful for exploring data where the groupings are not known in advance.

### How it works

The algorithm starts by placing `k` centroids at random positions, then repeats two steps until the centroids stabilize. First, it computes the [Euclidean distance](<doc:Linear-Algebra-Primer>) from each data point to every centroid and assigns each point to the nearest one. Second, it recomputes each centroid as the **mean** position of all points assigned to its cluster. This cycle continues until the centroids stop moving (convergence) or the maximum number of iterations is reached. 

![Scatter plot of points grouped into colored clusters with each cluster's centroid marked](diagram-kmeans)

Because initial centroid positions are random, different starting positions can produce different clusterings — the `seed` parameter ensures reproducible results. The `bestFit` method runs multiple initializations automatically and returns the model with the lowest inertia, avoiding poor outcomes caused by unlucky starting positions.

> Note: A **centroid** is simply the average position of a group of points. If three customers have spending scores of [10, 20, 30] and incomes of [40, 50, 60], their centroid is [20, 50] — the mean of each column. K-Means uses centroids to represent the "center" of each cluster.

### The distance connection

At its core, K-Means relies on the same `distance(to:)` operation used throughout Quiver's vector mathematics. This is Euclidean distance — the straight-line distance between two points in n-dimensional space, computed as √Σ(aᵢ − bᵢ)². The same function powers nearest-neighbor search in `KNearestNeighbors` and similarity operations in <doc:Similarity-Operations>. 

> Note: Distance builds on vector subtraction — each (aᵢ − bᵢ) term is one element of the difference vector. For a deeper look at how vector arithmetic works geometrically, see [Vectors](https://waynewbishop.github.io/swift-algorithms/20-vectors.html) in Swift Algorithms & Data Structures.

### Fitting a model

The `fit(data:k:maxIterations:seed:)` static method runs the iterative algorithm and returns a trained model with final centroids and cluster assignments:

```swift
import Quiver

// 6 points that form two natural groups
let data: [[Double]] = [
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
    [8.0, 8.0], [8.5, 7.5], [9.0, 8.5]
]

// Fit with 2 clusters — seed ensures reproducible results
let model = KMeans.fit(data: data, k: 2, seed: 42)
print(model)           // KMeans: 2 clusters, 6 points, converged in N iterations
print(model.labels)    // [0, 0, 0, 1, 1, 1] — individual properties still accessible
```

> Experiment: **The Quiver Notebook** is the right surface for sweeping the cluster count. Re-fit the snippet with k from 2 to 6, then change the seed and re-run — when the cluster assignments shift between seeds, the data does not yet have k natural groups. See <doc:Quiver-Notebook>.

Because centroids start at random positions, a single run can converge on a poor clustering — two groups split left-right instead of top-bottom, for example. The `bestFit` method solves this by running the algorithm multiple times, each with a different seed, and returning the model with the lowest inertia:

```swift
import Quiver

// Run 10 initializations, return the model with lowest inertia
let model = KMeans.bestFit(data: data, k: 2, attempts: 10)
```

### Inspecting results

After fitting, the model exposes everything needed to understand the clustering:

```swift
import Quiver

// Which cluster each point belongs to
print(model.labels)      // [0, 0, 0, 1, 1, 1]

// The center of each cluster
print(model.centroids)   // [[1.23, 1.97], [8.5, 8.0]]

// How tight the clusters are (lower = better)
print(model.inertia)     // sum of squared distances to centroids

// How many iterations until convergence
print(model.iterations)  // typically 2-10 for well-separated data
```

The `clusters(from:)` method groups data points by their assigned centroid, returning an array of `Cluster` values that conform to `Sequence`. This provides a natural way to iterate over the results:

```swift
import Quiver

// Group the training data into Cluster values
let clusters = model.clusters(from: data)

for cluster in clusters {
    print(cluster)  // Cluster: center [1.23, 1.97], 3 points
    for point in cluster {
        print("  \(point)")
    }
}
```

Each `Cluster` holds a `centroid`, the `points` assigned to it, and a `count`. Because `Cluster` conforms to `Sequence`, standard Swift patterns like `for-in` loops, `map`, and `filter` work directly on the cluster's data points.

### Predicting new points

The `predict(_:)` method assigns new data points to the nearest centroid without retraining:

```swift
import Quiver

// Assign new points to the nearest existing centroid
let newPoints: [[Double]] = [[2.0, 2.5], [7.0, 7.0]]
let assignments = model.predict(newPoints)
// [0, 1] — first point near cluster 0, second near cluster 1
```

### Choosing k

The parameter `k` determines how many clusters the algorithm creates. Each cluster is defined by a centroid — a point in vector space that represents the center of the group. Every data point is assigned to the cluster whose centroid is closest. Choosing too few clusters forces dissimilar points together; choosing too many splits natural groups apart.

The right number of clusters depends on the data. A common approach is the **elbow method** — fit models with increasing `k` and plot inertia. The "elbow" where inertia stops decreasing sharply suggests a good `k`:

```swift
import Quiver

let data: [[Double]] = [
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
    [5.0, 5.0], [5.5, 4.8], [4.8, 5.2],
    [9.0, 8.0], [8.5, 8.5], [9.2, 7.8]
]

// Fit models with increasing k and compare inertia
for k in 1...5 {
    let model = KMeans.fit(data: data, k: k, seed: 42)
    print("k=\(k): inertia=\(model.inertia)")
}
// k=1: inertia=~120   (big drop ahead)
// k=2: inertia=~30    (big drop ahead)
// k=3: inertia=~2     ← elbow
// k=4: inertia=~1.5   (diminishing returns)
// k=5: inertia=~0.8
```

### The full pipeline

A typical workflow combines feature scaling, clustering, and analysis:

```swift
import Quiver

// Customer data: spending score, annual income (different scales)
let data: [[Double]] = [
    [15.0, 39000], [16.0, 81000], [17.0, 6000],
    [81.0, 77000], [77.0, 40000], [76.0, 76000],
    [94.0, 3000],  [87.0, 72000], [90.0, 88000]
]

// Scale features so distance treats both equally
let scaler = StandardScaler.fit(features: data)
let scaled = scaler.transform(data)

// Cluster
let model = KMeans.fit(data: scaled, k: 3, seed: 42)

// Inspect cluster sizes
for cluster in model.clusters(from: scaled) {
    print("Center: \(cluster.centroid), customers: \(cluster.count)")
}
```

### Organizing data with Panel

The same pipeline using `Panel` keeps column names attached to the data throughout:

```swift
import Quiver

let customers = Panel([
    ("spending", [15.0, 16.0, 17.0, 81.0, 77.0, 76.0, 94.0, 87.0, 90.0]),
    ("income",   [39.0, 81.0, 6.0, 77.0, 40.0, 76.0, 3.0, 72.0, 88.0])
])

let features = customers.toMatrix(columns: ["spending", "income"])
let scaler = StandardScaler.fit(features: features)
let model = KMeans.fit(data: scaler.transform(features), k: 3, seed: 42)

print(model.labels)
```

`Panel` is entirely optional. The clustering algorithm accepts arrays directly, and developers who prefer working with raw arrays can continue to do so. See <doc:Panel> for details.

### When to use K-Means

K-Means works best when clusters are roughly spherical and similarly sized, the number of clusters is known or can be estimated, and features are continuous and scaled to similar ranges. `StandardScaler` is the recommended choice for distance-based algorithms because it centers each feature at zero with unit variance, preventing high-magnitude features from dominating the distance calculation. `FeatureScaler` (min-max scaling) is an alternative when a bounded [0, 1] range is preferred. It is a natural fit for customer segmentation, anomaly detection, and exploring structure in unlabeled data.

K-Means struggles with non-spherical cluster shapes (elongated, curved, or nested), clusters of very different sizes or densities, and categorical data. It also cannot determine the "right" `k` automatically — the elbow method helps, but the final choice requires domain knowledge. Note that K-Means assigns every point to a cluster — there are no unassigned outliers, so unusual data points will be forced into the nearest group.

### Safe by design

`KMeans` follows the same immutable-struct pattern as `GaussianNaiveBayes`, `LinearRegression`, and `KNearestNeighbors`. The model is always ready to use after `fit`, the training data stays separate from the result, and reproducible seeds ensure consistent results across runs.

Both `KMeans` and `Cluster` conform to Swift's `Equatable` protocol. When two runs use the same data and the same seed, the results are guaranteed identical:

```swift
import Quiver

let run1 = KMeans.fit(data: points, k: 3, seed: 42)
let run2 = KMeans.fit(data: points, k: 3, seed: 42)
run1 == run2  // true

let clusters1 = run1.clusters(from: points)
let clusters2 = run2.clusters(from: points)
clusters1 == clusters2  // true
```

This is useful for unit tests, debugging, and verifying that a pipeline produces stable output.

## Topics

### Model
- ``KMeans``
- ``Cluster``

### Training
- ``KMeans/fit(data:k:maxIterations:seed:)``
- ``KMeans/bestFit(data:k:maxIterations:attempts:)``

### Prediction
- ``KMeans/predict(_:)``
- ``KMeans/clusters(from:)``

### Related
- <doc:Machine-Learning-Primer>
- <doc:Nearest-Neighbors-Classification>
- <doc:Feature-Scaling>
- <doc:Pipeline>
