# Vector Operations

Transform arrays into mathematical vectors with magnitude, direction, and spatial relationships.

## Overview

Quiver provides a comprehensive set of vector operations that treat `arrays` as mathematical **vectors**, enabling calculations like `magnitude`, `normalization`, dot products, and angle measurements. For the conceptual foundations behind these operations, see <doc:Linear-Algebra-Primer>. 

> Tip: Vector operations are essential for graphics programming, physics simulations, machine learning algorithms, and any application that deals with spatial data or mathematical modeling. For a comprehensive introduction to vector mathematics, see [Vectors](https://waynewbishop.github.io/swift-algorithms/20-vectors.html) in Swift Algorithms & Data Structures.

### Basic vector properties

Quiver enables calculating fundamental vector properties. A `vector` is a mathematical object that represents both `magnitude` and direction. Unlike a scalar value, which represents only size (like temperature or weight), a `vector` captures directional information alongside its size:

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

These angle functions work with vectors of any dimension. The `cosineOfAngle(with:)` method returns the raw cosine value, while `angle(with:)` applies `acos()` to produce the angle in radians.

> Tip: The dot product is zero when vectors are perpendicular.

### Distance

Both `magnitude` and `distance(to:)` use the Pythagorean theorem, but they measure different things. `magnitude` measures a single vector's length from the origin — how far a point is from `[0, 0, ...]`. `distance(to:)` measures the gap between any two points by computing the `magnitude` of their difference vector:

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

This distinction matters throughout Quiver. Cosine similarity divides by both magnitudes to remove length bias and isolate direction. Quiver's models use `distance(to:)` to find the most similar training examples, group data points into clusters, and rank how related two arrays are. For a deeper look at how distance and similarity work together, see <doc:Similarity-Operations>.

### Vector arithmetic

Quiver provides `add`, `subtract`, `multiply`, and `divide` methods for element-wise array arithmetic. These methods are the foundation that higher-level vector operations build on:

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

These methods appear throughout Quiver's ML pipeline. `distance(to:)` is implemented as `self.subtract(other).magnitude` — it subtracts two vectors element-wise, then computes the `magnitude` of the difference. Every time `KNearestNeighbors` finds the closest training example or `KMeans` assigns a point to a cluster, it relies on this subtraction:

```swift
let sample = [5.2, 3.1]
let trainingPoint = [4.8, 3.5]

// distance(to:) subtracts, then takes magnitude
let diff = sample.subtract(trainingPoint)  // [0.4, -0.4]
diff.magnitude                          // √(0.16 + 0.16) ≈ 0.566
sample.distance(to: trainingPoint)      // 0.566 (same result)
```

Addition and division power `averaged()`, which combines multiple word embedding vectors into a single document vector for semantic search. Individual word vectors each capture one word's meaning — averaging them produces a vector that represents the entire document's meaning in the same vector space:

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

> Tip: For a complete walkthrough of the embedding-to-search pipeline, see <doc:Semantic-Search>.

Subtraction also gives displacement — the vector from one point to another. A player at `[100, 200]` and an enemy at `[130, 170]` have displacement `[100, 200].subtract([130, 170])` = `[-30, 30]`. The `magnitude` of that displacement is the distance between them. Addition combines forces or velocities — a boat moving at `[3, 0]` in a current of `[0, 2]` has actual velocity `[3, 0].add([0, 2])` = `[3, 2]`.

> Important: The `multiply(_:)` method performs element-wise multiplication (Hadamard product), not matrix multiplication. For matrix multiplication, use `multiplyMatrix()`.

### Matrix-vector operations

Transform vectors using matrices:

```swift
let vector = [1.0, 2.0]
let matrix = [[0.0, -1.0], [1.0, 0.0]]  // 90° rotation matrix

// Apply matrix transformation (two equivalent ways)
let transformed = vector.transformedBy(matrix)  // [-2.0, 1.0]
let transformed2 = matrix.transform(vector)     // [-2.0, 1.0]
```

> Tip: Use `matrix.transform(vector)` to emphasize the matrix acting on the vector, matching mathematical notation Mv = w. Use `vector.transformedBy(matrix)` to emphasize the vector being transformed.

Matrix transformations are powerful tools for implementing rotations, scaling, and other geometric operations.

### Mathematical foundation

Vector operations in Quiver are based on well-established mathematical principles:

- **Magnitude**: √(x₁² + x₂² + ... + xₙ²)
- **Normalization**: v / ||v||
- **Dot product**: v₁·v₂ = v₁₁×v₂₁ + v₁₂×v₂₂ + ... + v₁ₙ×v₂ₙ
- **Euclidean distance**: d(v₁, v₂) = √((v₁₁−v₂₁)² + (v₁₂−v₂₂)² + ... + (v₁ₙ−v₂ₙ)²)
- **Cosine similarity**: cos(θ) = (v₁·v₂) / (||v₁|| × ||v₂||)

> Note: Quiver follows standard mathematical conventions for vector operations, making it easier to translate mathematical formulas directly into code.

## Topics

### Vector properties
- ``Swift/Array/magnitude``
- ``Swift/Array/normalized``

### Vector Relationships
- ``Swift/Array/distance(to:)``
- ``Swift/Array/dot(_:)``
- ``Swift/Array/angle(with:)-piry``
- ``Swift/Array/angleInDegrees(with:)-7n2tx``
- ``Swift/Array/cosineOfAngle(with:)``
- ``Swift/Array/averaged()``

### Matrix operations
- ``Swift/Array/transformedBy(_:)``

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
