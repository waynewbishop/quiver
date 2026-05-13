# Composing Transformations

Combining transformations with matrix multiplication for graphics pipelines and animation systems.

## Overview

Individual transformations like rotation and scaling are useful, but real applications often require combining them. Matrix multiplication folds several transformations into a single matrix that applies in one pass over the vector.

That folding shows up in graphics pipelines, animation systems, and any application that chains coordinate system changes. For the individual matrix forms — rotation, scaling, reflection, and shear — see <doc:Common-Transformations>. This article focuses on how those matrices combine.

> Note: For geometric intuition about how transformations compose, including visual examples of rotation, scaling, and chained operations, see [Matrix Transformations](https://waynewbishop.github.io/swift-algorithms/22-matrix-transformations.html) in Swift Algorithms & Data Structures.

## Matrix multiplication

Matrix multiplication composes transformations: the result represents applying one transformation after another. Unlike scalar multiplication, the order is significant — `A × B` is generally different from `B × A`.

```swift
import Quiver

// Rotation and scaling matrices from <doc:Common-Transformations>
let rotation = [
    [0.707, -0.707],
    [0.707,  0.707]
]
let scaling = [Double].diag([2.0, 2.0])

// Multiply matrices to compose transformations
let combined = rotation.multiplyMatrix(scaling)

// Apply to vector
let v = [1.0, 0.0]
let result = v.transformedBy(combined)
```

### Two equivalent approaches

Composition gives us a choice between precomputing the combined matrix and applying transformations sequentially. Both produce the same result.

```swift
// Compose first, apply once
let combined = transform1.multiplyMatrix(transform2)
let result = vector.transformedBy(combined)

// Apply sequentially
let result = vector
    .transformedBy(transform1)
    .transformedBy(transform2)
```

The composed-matrix form is more efficient when applying the same transformation to many vectors, because the matrix multiplication runs once instead of once per vector.

## Order matters

Matrix multiplication is not commutative. Reversing the order of two transformations generally produces a different combined matrix and a different final result.

```swift
let rotate90 = [[0.0, -1.0], [1.0,  0.0]]
let scale2x  = [Double].diag([2.0, 1.0])

let v = [1.0, 0.0]

// Rotate then scale: [1,0] → rotate → [0,1] → scale → [0,1]
let rotateFirst = v.transformedBy(rotate90).transformedBy(scale2x)

// Scale then rotate: [1,0] → scale → [2,0] → rotate → [0,2]
let scaleFirst = v.transformedBy(scale2x).transformedBy(rotate90)

rotateFirst // [0, 1]
scaleFirst  // [0, 2]
```

### Reading order

When composing with matrix multiplication, the rightmost matrix is applied first.

```swift
let combined = A.multiplyMatrix(B)
```

This means: first apply B, then apply A. The convention follows directly from how matrix-vector multiplication associates — `(A × B) × v` evaluates as `A × (B × v)`, with B acting on the vector before A.

## Common composition patterns

### Scale then rotate

Scaling along the axes is straightforward when those axes are aligned with x and y. Rotation changes the orientation of the coordinate system, which makes any subsequent axis-aligned scaling more complex. The general guideline is to scale and shear first, then rotate last.

```swift
// Compose: scale first, then rotate (rightmost applied first)
let scale = [Double].diag([2.0, 2.0])
let rotate = [[0.707, -0.707], [0.707, 0.707]]

let scaleRotate = rotate.multiplyMatrix(scale)
```

### Rotate around a point

Rotation matrices rotate vectors around the origin. To rotate around any other point, the standard pattern is a three-step sequence: translate so the pivot lands at the origin, rotate, and translate back.

```swift
let pivot = [5.0, 5.0]
let vector = [6.0, 5.0]
let rotate90 = [[0.0, -1.0], [1.0, 0.0]]

// 1. Shift to origin, 2. Rotate, 3. Shift back
let rotated = vector.subtract(pivot)
    .transformedBy(rotate90)
    .add(pivot)
// Result: [5.0, 6.0]
```

### Combining different transformations

Scaling, shearing, and rotation can be composed into a single matrix that captures all three operations. The rightmost matrix is applied first.

```swift
let scale  = [Double].diag([2.0, 1.5])
let shear  = [[1.0, 0.3], [0.0, 1.0]]
let rotate = [[0.0, -1.0], [1.0, 0.0]]

// Compose (rightmost applied first): scale → shear → rotate
let complex = rotate.multiplyMatrix(shear).multiplyMatrix(scale)

// Apply to many vectors efficiently
let vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
let transformed = vectors.map { $0.transformedBy(complex) }
```

## Transformation pipelines

Graphics applications commonly use transformation pipelines that move a vertex through several coordinate spaces — object to world, world to camera, camera to screen. Each space is reached by composing a matrix onto the previous one.

```swift
// Object → World → Camera → Screen
let objectToScreen = cameraToScreen
    .multiplyMatrix(worldToCamera)
    .multiplyMatrix(objectToWorld)

// Transform all vertices once
let screenVertices = objectVertices.map { $0.transformedBy(objectToScreen) }
```

### Incremental updates

When only one stage of the pipeline changes — usually because an object moves while the camera and projection remain fixed — caching the partial composition saves redundant work.

```swift
// Cache partial compositions
let worldToScreen = cameraToScreen.multiplyMatrix(worldToCamera)

// When object moves, only update final step
let newObjectToScreen = worldToScreen.multiplyMatrix(newObjectTransform)
```

## Inverse transformations

Some transformations can be reversed by composing them with an inverse matrix that undoes their effect. A 90° rotation followed by a -90° rotation returns to the identity, and a 2× scale followed by a 0.5× scale does the same.

```swift
let rotate   = [[0.0, -1.0], [1.0,  0.0]]
let unrotate = [[0.0,  1.0], [-1.0, 0.0]]

let identity = rotate.multiplyMatrix(unrotate)
// ≈ [[1,0],[0,1]] (within floating-point precision)
```

## Performance optimization

When the same chain of transformations is applied to many vectors, composing the matrices once and applying the result repeatedly is significantly faster than applying each matrix individually to every vector. Matrix multiplication is `O(n³)` for n×n matrices; matrix-vector multiplication is `O(n²)`. Composing the chain once trades a single `O(n³)` operation for many cheaper `O(n²)` operations.

```swift
// Transform 1000 vertices
let vertices: [[Double]] = // 1000 2D points

// Inefficient: 1000 × (2 matrix-vector) = 12,000 operations
let result1 = vertices.map { $0.transformedBy(A).transformedBy(B) }

// Efficient: 1 matrix-matrix + 1000 matrix-vector = 6,012 operations
let AB = A.multiplyMatrix(B)
let result2 = vertices.map { $0.transformedBy(AB) }
```

> Experiment: **The Quiver Notebook** is the right place to feel why order matters in matrix composition. Take `rotate90` and `scale2x` from the example above and compose them both ways — `rotate.multiplyMatrix(scale)` and `scale.multiplyMatrix(rotate)`. Apply each composed matrix to the same vector `[1.0, 0.0]` and print the results side by side. The two final positions are different, and seeing them next to each other is the fastest way to internalize that matrix multiplication is not commutative. See <doc:Quiver-Notebook>.

## Topics

### Transformation operations
- ``Swift/Array/transformedBy(_:)``

### Matrix creation
- ``Swift/Array/identity(_:)``
- ``Swift/Array/diag(_:)``
