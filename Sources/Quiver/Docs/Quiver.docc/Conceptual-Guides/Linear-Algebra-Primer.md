# Linear Algebra Primer

Understand the math behind Quiver's vector operations and machine learning models.

## Overview

Quiver treats Swift `Array` types as mathematical objects, computing `magnitude`, measuring angles between them, and transforming coordinates with matrices. These operations come from **linear algebra**, the branch of mathematics that deals with vectors and the transformations that act on them.

Linear algebra is also the mathematical foundation of machine learning. Quiver's ML models are built from concepts introduced in this primer.

> Note: Advanced mathematical knowledge is not required. Working with Swift arrays is enough to start applying these concepts in code.

### Arrays are vectors

As programmers, we work with `Array` types constantly, storing coordinates, pixel values, sensor readings, and feature scores. In linear algebra, these same arrays are called **vectors**. The difference is not in the data structure but in what we can do with it. Once we treat an `Array` as a vector, we gain access to operations that measure its length, its direction, and its relationship to other arrays.

> Tip: For details on how Quiver extends the Swift `Array` see <doc:How-It-Works>.

Consider the arrays we already use every day. An array tracking wind speed and bearing captures both "how much" and "which way". An array of RGB values describes a color as a point in three-dimensional space. An array of customer preferences across product features places that customer at a specific location in a multidimensional space. These can all be considered vectors.

Quiver bridges this gap by adding vector operations directly to Swift `Array` types. No new types to learn and no conversion step. The arrays we already have gain mathematical capabilities:

```swift
import Quiver

let v = [3.0, 4.0]

// Magnitude: the length of the vector
v.magnitude  // 5.0

// Direction: a unit vector (length of 1) pointing the same way
v.normalized  // [0.6, 0.8]

// See the rational form
v.normalized.asFractions()  // [3/5, 4/5]
```

The `magnitude` property is calculated using the Pythagorean theorem extended to any dimension. For a 2D vector `[x, y]`, magnitude equals √(x² + y²). For vector `[3, 4]`, that gives √(9 + 16) = √25 = 5. The same formula works whether the vector has 2 dimensions or 200.

**Normalization** separates "how much" from "which way" by dividing each element by the `magnitude`. The result is a **unit vector** — an `Array` with length 1 that preserves only the direction of the original. This matters when comparing arrays where scale varies but direction is what we care about. Two customer profiles with the same preference ratios but different spending levels point in the same direction. Normalization reveals that similarity by removing the magnitude.

### Vectors live in space

Every vector has a position in **vector space** — a coordinate system where each element in the `Array` represents a dimension. A 2D vector is a point on a flat plane. A 3D vector is a point in a cube. An `Array` with hundreds of elements is a point in a space we can't visualize, but the math works exactly the same way.

This is the key idea behind all of Quiver's operations. Vector space is what makes it possible to treat completely different things — flowers, documents, products, sensor readings — with the same mathematics. As long as we can describe something as an `Array` of numbers, it has a position in vector space. And once something has a position, we can measure how far it is from anything else — which is exactly what machine learning algorithms do.

### Distance, direction, and meaning

Two product feature vectors that are close together in vector space represent similar products. A cluster of data points near each other form a natural group. The math doesn't care what the numbers represent — it only cares about position, distance, and direction.

This is also how semantic search works. Words can be represented as high-dimensional vectors called **embeddings** — arrays of hundreds of numbers that capture meaning. The word "running" and the word "jogging" end up near each other in vector space because they appear in similar contexts. Searching for related content becomes a matter of finding the nearest vectors using `cosineOfAngle(with:)`. Quiver's <doc:Semantic-Search> page walks through building this from scratch.

### The dot product

The **dot product** is the fundamental building block for measuring similarity. It takes two vectors and produces a single number, the sum of their element-wise products, that reflects how much the vectors agree. When two vectors point in the same direction, the dot product is large and positive. When they are perpendicular, it equals zero. When they point in opposite directions, it is negative.

```swift
let v1 = [1.0, 0.0]  // Points right
let v2 = [0.0, 1.0]  // Points up

// Perpendicular vectors: dot product is zero
v1.dot(v2)  // 0.0

// Parallel vectors: dot product equals the product of their magnitudes
v1.dot(v1)  // 1.0
```

The dot product appears throughout Quiver — in similarity measurements, matrix transformations, and directly inside the ML models. It is one of the most frequently used operations in numerical computing.

### Cosine similarity

The dot product gives a general signal of similarity, but its value depends on magnitude. Two vectors pointing in identical directions but with different lengths produce vastly different dot products. Cosine similarity fixes this by dividing the dot product by both vectors' magnitudes, canceling out length and measuring only the angle between them. The result falls between -1 and 1, where 1 means identical direction, 0 means unrelated, and -1 means opposite:

```swift
// Feature vectors for two products
let product1 = [4.2, 7.8, 3.1, 9.5]
let product2 = [3.8, 8.2, 2.9, 9.7]

// Cosine similarity: direction only, magnitude ignored
product1.cosineOfAngle(with: product2)  // ~0.999 (very similar)
```

This separation of direction from magnitude is why cosine similarity powers recommendation engines, search ranking, and duplicate detection. Two customer profiles with identical preferences but different engagement levels point in the same direction. Cosine similarity scores them as nearly identical, while the raw dot product would not.

> Tip: Learn the mathematics behind dot product, cosine similarity, and matrix multiplication in [Swift Algorithms & Data Structures](https://waynewbishop.github.io/swift-algorithms/20-vectors.html).

### What matrices do

While vectors represent individual points or directions in space, **matrices** provide the framework for transforming them. A matrix is a rectangular grid of numbers that describes a rule for moving vectors to new positions. Consider a rotation:

```swift
// A rotation matrix
let rotation = [
    [0.0, -1.0],
    [1.0,  0.0]
]

// A point in 2D space
let point = [3.0, 1.0]

// Apply the rotation
// row 0: (0.0 * 3.0) + (-1.0 * 1.0) = -1.0
// row 1: (1.0 * 3.0) + ( 0.0 * 1.0) =  3.0
point.transformedBy(rotation)  // [-1.0, 3.0]
```

The point `[3.0, 1.0]` moves to `[-1.0, 3.0]`. The matrix describes the rule; `transformedBy` applies it. This particular matrix doesn't represent data about an object — it represents an *operation* that rotates any vector to a new position. Matrices can also scale (stretch or compress), reflect (mirror across an axis), shear (tilt), and compose multiple transformations together.

Beyond transformations, matrices organize collections of data. In a dataset, each row might represent a different sample and each column a different measurement. A matrix of athlete performance data with rows for athletes and columns for speed, endurance, and strength is three vectors stacked together — and matrix operations let us process all of them simultaneously.

> Important: For a matrix to transform a vector, the number of columns must match the vector's length.

The same machinery solves systems of linear equations. Given the system `Ax = b`, where `A` is a square matrix and `b` is a known right-hand side, the unknown vector `x` satisfies `x = A⁻¹b`. Quiver exposes this directly as `solve(_:)`:

```swift
// 2x +  y = 5
//  x + 3y = 10
let A = [[2.0, 1.0],
         [1.0, 3.0]]
let b = [5.0, 10.0]
A.solve(b)   // [1.0, 3.0]
```

The method returns `nil` when the matrix is singular — the same condition that makes inversion fail. For the geometric meaning of singularity and condition number, see <doc:Determinants-Primer>.

### From arrays to algorithms

Linear algebra isn't just math exercises — these concepts are the building blocks for ML models in Quiver. The key concept connecting them is **distance** — the measurement of how far apart two points sit in vector space.

Both `magnitude` and `distance(to:)` use the Pythagorean theorem, but they measure different things. Think of `magnitude` as the answer to "how far is this point from the origin" — it measures a single vector's length. `distance(to:)` answers a broader version of the same question — how far apart are any two points — by subtracting one vector from the other and computing the length of the difference.

```swift
let a = [1.0, 2.0]
let b = [4.0, 6.0]

// Magnitude: distance from origin
a.magnitude  // √(1² + 2²) = √5 ≈ 2.24

// Euclidean distance: distance between two points
a.distance(to: b)  // √((4-1)² + (6-2)²) = √25 = 5.0
```

> Tip: Distance builds on vector subtraction — each (aᵢ − bᵢ) term is one element of the difference vector. The `magnitude` of that difference vector is the distance between the two points.

`Distance` is what connects linear algebra to machine learning. Quiver's models use distance to find the most similar training examples, group data points together, and rank how related two arrays are. The <doc:Machine-Learning-Primer> explores each of these models and how they apply these concepts.

> Tip: For a course teaching applied linear algebra in Swift, the <doc:Quiver-Notebook-For-Classrooms> page covers the classroom adoption model — every student runs the same environment from one `swift run`, with no per-machine setup.

