# Vector Operations

Give arrays magnitude, direction, and distance by treating them as mathematical vectors.

## Overview

Quiver treats ordinary `arrays` as mathematical **vectors**, so a plain list of numbers gains a length, a direction, and a set of operations for comparing it to other vectors. From an array we can compute its `magnitude`, produce a normalized version, take a dot product, or measure the angle to another vector. For the ideas behind these operations, see <doc:Linear-Algebra-Primer>.

> Note: Vector operations underpin machine learning, and they turn up anywhere an app reasons about direction, distance, or spatial data. For a comprehensive introduction to vector mathematics, see [Vectors](https://waynewbishop.github.io/swift-algorithms/20-vectors.html) in Swift Algorithms & Data Structures.

### Basic vector properties

Every vector has two fundamental properties: a `magnitude` (its length) and a direction (where it points). This is what sets a vector apart from a scalar. A scalar is a single number that carries only size — a temperature or a weight — while a vector carries a direction as well. Each number inside a vector is a component, and the count of components is the vector's dimension: `[3.0, 4.0]` has two components, so it lives in two dimensions.

```swift
import Quiver

let vector = [3.0, 4.0]

// Calculate the magnitude (length) of the vector
vector.magnitude  // 5.0

// Create a normalized version (unit vector)
vector.normalized  // [0.6, 0.8]

// See the rational form
vector.normalized.asFractions()  // [3/5, 4/5]
```

> Important: Calling `normalized` on a zero vector returns a zero vector.

### Vector relationships

Calculate relationships between vectors with these operations:

```swift
let v1 = [3.0, 0.0]  // Vector along x-axis
let v2 = [0.0, 4.0]  // Vector along y-axis

// Calculate the dot product
v1.dot(v2)  // 0.0 (perpendicular vectors)

// Calculate the angle between vectors
v1.angle(with: v2)        // π/2 radians (90 degrees)
v1.angleInDegrees(with: v2)  // 90.0 degrees
```

These angle functions work with vectors of any dimension. The `cosineOfAngle(with:)` method returns the cosine of the angle directly, and `angle(with:)` applies `acos` to that cosine to give the angle in radians. This is why perpendicular vectors are a clean test case: their dot product is zero, so the cosine is zero, and the angle comes out to 90°.

> Experiment: **The Quiver Notebook** is the right place to watch the angle change. Hold one vector fixed and rotate the other from aligned, to perpendicular, to pointing the opposite way — the cosine of the angle moves through 1.0 → 0 → −1. A single number tells us how closely the two vectors point in the same direction. See <doc:Quiver-Notebook>.

### Distance

`magnitude` and `distance(to:)` both rest on the Pythagorean theorem, but they answer different questions. `magnitude` measures one vector's length from the origin — how far a point sits from `[0, 0, ...]`. `distance(to:)` measures the gap between any two points, which it finds by taking the `magnitude` of the vector between them:

```swift
let a = [1.0, 2.0]
let b = [4.0, 6.0]

// Magnitude: distance from origin
a.magnitude  // √(1² + 2²) = √5 ≈ 2.24

// Euclidean distance: distance between two points
a.distance(to: b)  // √((4-1)² + (6-2)²) = √25 = 5.0

// Magnitude is the special case — distance from origin
[0.0, 0.0].distance(to: a)  // √5 ≈ 2.24 (same as a.magnitude)
```

This distinction matters throughout Quiver. Its models use `distance(to:)` to find the closest training examples, to group data points into clusters, and to rank how related two arrays are. Measuring the angle between vectors is a separate tool. Dividing the dot product by both magnitudes cancels length and leaves a pure measure of direction. For how distance and angle-based comparison work together, see <doc:Similarity-Operations>.

### Choosing a distance metric

The straight line is one of three ways to measure the same gap. `distance(to:metric:)` accepts a ``DistanceMetric`` and answers a different question with each case. **Manhattan distance** sums the steps along each axis, and **cosine distance** compares direction while ignoring magnitude:

```swift
a.distance(to: b, metric: .euclidean)  // 5.0 (straight line)
a.distance(to: b, metric: .manhattan)  // |4−1| + |6−2| = 7.0 (steps along each axis)
a.distance(to: b, metric: .cosine)     // 0.0077 (direction, not amount)
```

The choice is a modeling decision. Manhattan distance is more robust to outliers than Euclidean, since a large difference in one feature contributes linearly rather than squared. Cosine distance suits text embeddings and other high-dimensional data where orientation carries the meaning. The `.cosine` case is `1 − cosineOfAngle(with:)`, the similarity score flipped so that smaller means closer; <doc:Similarity-Operations> walks through the inversion. The same `metric:` parameter appears when fitting a classifier in <doc:Nearest-Neighbors-Classification>, so a metric explored here behaves identically inside the model.

### Vector arithmetic

Quiver provides `add`, `subtract`, `multiply`, and `divide` methods for element-wise array arithmetic — each operation pairs up matching components and combines them. These methods are the foundation that higher-level vector operations build on:

```swift
import Quiver

let a = [1.0, 2.0, 3.0]
let b = [4.0, 5.0, 6.0]

// Element-wise operations
let sum = a.add(b)            // [5.0, 7.0, 9.0]
let difference = a.subtract(b)  // [-3.0, -3.0, -3.0]
let product = a.multiply(b)     // [4.0, 10.0, 18.0]
let quotient = a.divide(b)      // [0.25, 0.4, 0.5]
```

These methods run underneath Quiver's machine learning models. `distance(to:)` is really just `self.subtract(other).magnitude` — it subtracts the two vectors component by component, then takes the `magnitude` of what's left. Every time `KNearestNeighbors` finds the closest training example, or `KMeans` assigns a point to a cluster, that subtraction is doing the work:

```swift
let sample = [5.2, 3.1]
let trainingPoint = [4.8, 3.5]

// distance(to:) subtracts, then takes magnitude
let diff = sample.subtract(trainingPoint)  // [0.4, -0.4]
diff.magnitude                          // √(0.16 + 0.16) ≈ 0.566
sample.distance(to: trainingPoint)      // 0.566 (same result)
```

Addition and division together power `averaged`, which combines several word vectors into a single vector. Each word vector captures the meaning of one word. Averaging them produces one vector that stands for the meaning of the whole document, in the same space as the words themselves:

```swift
// Word embedding vectors (simplified to 3 dimensions)
let wordVectors = [
    [0.8, 0.2, 0.1],   // "running"
    [0.7, 0.3, 0.2],   // "athletic"
    [0.6, 0.1, 0.3],   // "shoes"
    [0.1, 0.6, 0.4]    // "comfortable"
]

// Average into a single document vector
if let documentVector = wordVectors.averaged() {
    print(documentVector)  // [0.55, 0.3, 0.25] — represents the full document
}
```

> Note: For a complete walkthrough of the embedding-to-search pipeline, see <doc:Semantic-Search>.

Subtraction also gives displacement: the vector from one point to another. A player at `[100, 200]` and an enemy at `[130, 170]` have displacement `[100, 200].subtract([130, 170])` = `[-30, 30]`. The `magnitude` of that displacement is the distance between them. Addition combines forces or velocities. A boat moving at `[3, 0]` in a current of `[0, 2]` has actual velocity `[3, 0].add([0, 2])` = `[3, 2]`.

> Important: The `multiply(_:)` method performs element-wise multiplication (Hadamard product), not matrix multiplication. For matrix multiplication, use `multiplyMatrix`.

### Matrix-vector operations

Transform vectors using matrices:

```swift
let vector = [1.0, 2.0]
let matrix = [[0.0, -1.0], [1.0, 0.0]]  // 90° rotation matrix

// Apply matrix transformation (two equivalent ways)
let transformed = vector.transformedBy(matrix)  // [-2.0, 1.0]
let transformed2 = matrix.transform(vector)     // [-2.0, 1.0]
```

> Note: Use `matrix.transform(vector)` to emphasize the matrix acting on the vector, matching mathematical notation Mv = w. Use `vector.transformedBy(matrix)` to emphasize the vector being transformed.

Matrix transformations are how we implement rotations, scaling, and other geometric operations.

### Mathematical foundation

Vector operations in Quiver are based on well-established mathematical principles:

- **Magnitude**: √(x₁² + x₂² + ... + xₙ²)
- **Normalization**: v / ||v||
- **Dot product**: v₁·v₂ = v₁₁×v₂₁ + v₁₂×v₂₂ + ... + v₁ₙ×v₂ₙ
- **Euclidean distance**: d(v₁, v₂) = √((v₁₁−v₂₁)² + (v₁₂−v₂₂)² + ... + (v₁ₙ−v₂ₙ)²)
- **Cosine of the angle**: cos(θ) = (v₁·v₂) / (||v₁|| × ||v₂||)

> Note: Quiver follows standard mathematical conventions for vector operations, making it easier to translate mathematical formulas directly into code.

## Topics

### Vector properties
- ``Swift/Array/magnitude``
- ``Swift/Array/normalized``

### Vector relationships
- ``Swift/Array/distance(to:)``
- ``Swift/Array/distance(to:metric:)->Double``
- ``Swift/Array/distance(to:metric:)->Element``
- ``Swift/Array/dot(_:)``
- ``Swift/Array/angle(with:)-piry``
- ``Swift/Array/angleInDegrees(with:)-7n2tx``
- ``Swift/Array/cosineOfAngle(with:)``
- ``Swift/Array/averaged()``

### Matrix operations
- ``Swift/Array/transformedBy(_:)``

### Rendering
- ``VectorForm``

### Related articles
- <doc:Vector-Projections>
- <doc:Linear-Algebra-Primer>
- <doc:Similarity-Operations>
- <doc:Boolean-Masking>
- <doc:Semantic-Search>
- <doc:Matrix-Transformations>
- <doc:Matrix-Operations>
- <doc:Fourier-Transform>
- <doc:Broadcasting-Operations>
