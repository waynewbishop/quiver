# Composing Transformations

Combining transformations with matrix multiplication for graphics pipelines and animation systems.

## Overview

Individual transformations like rotation and scaling are useful, but real applications often require combining them. Matrix multiplication lets us compose multiple transformations into a single operation, producing complex effects efficiently.

Understanding transformation composition is fundamental to graphics pipelines, animation systems, and any application that chains coordinate system changes.

> Tip: For geometric intuition about how transformations compose, including visual examples of rotation, scaling, and chained operations, see [Matrix Transformations](https://waynewbishop.github.io/swift-algorithms/22-matrix-transformations.html) in Swift Algorithms & Data Structures.

## Matrix multiplication

Matrix multiplication composes transformations: the result represents applying one transformation after another. Unlike scalar multiplication, the order is significant — `A × B` is generally different from `B × A`.

```swift
import Quiver

// 45° counterclockwise rotation and scaling matrices
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
import Foundation

// 90° counterclockwise rotation and scaling matrices
let rotate90 = [
    [0.0, -1.0],
    [1.0,  0.0]
]
let scale2x = [Double].diag([2.0, 1.0])

let v = [1.0, 0.0]

// Rotate then scale
let rotateFirst = v.transformedBy(rotate90).transformedBy(scale2x)
// [1,0] → rotate → [0,1] → scale → [0,1]

// Scale then rotate
let scaleFirst = v.transformedBy(scale2x).transformedBy(rotate90)
// [1,0] → scale → [2,0] → rotate → [0,2]

rotateFirst // [0, 1]
scaleFirst  // [0, 2]
```

### Reading order

When composing with matrix multiplication, the rightmost matrix is applied first.

```swift
let combined = A.multiplyMatrix(B)
```

This means: first apply B, then apply A. The convention follows directly from how matrix-vector multiplication associates — `(A × B) × v` evaluates as `A × (B × v)`, with B acting on the vector before A.

```swift
// Rotate then scale
let combined = scale.multiplyMatrix(rotate)
v.transformedBy(combined)
// Same as: v.transformedBy(rotate).transformedBy(scale)
```

## Common composition patterns

### Scale then rotate

Scaling along the axes is straightforward when those axes are aligned with x and y. Rotation changes the orientation of the coordinate system, which makes any subsequent axis-aligned scaling more complex. The general guideline is to scale and shear first, then rotate last.

```swift
// Make sprite 2× larger, then rotate 45°
let scale = [Double].diag([2.0, 2.0])

// 45° counterclockwise rotation
let rotate = [
    [0.707, -0.707],
    [0.707,  0.707]
]

// Compose: scale first, then rotate
let scaleRotate = rotate.multiplyMatrix(scale)
```

### Rotate around a point

Rotation matrices rotate vectors around the origin. To rotate around any other point, the standard pattern is a three-step sequence: translate so the pivot lands at the origin, rotate, and translate back.

```swift
// Rotate around pivot point using a 3-step sequence
let pivot = [5.0, 5.0]
let vector = [6.0, 5.0]

// 90° rotation: [[0, -1], [1, 0]]
let rotate90 = [
    [0.0, -1.0],
    [1.0,  0.0]
]

// 1. Shift to origin, 2. Rotate, 3. Shift back
let rotated = vector.subtract(pivot)
    .transformedBy(rotate90)
    .add(pivot)
// vector.subtract(pivot) = [1, 0]
// Row 1: [0, -1] • [1, 0] = (0×1 + (-1)×0) = 0
// Row 2: [1,  0] • [1, 0] = (1×1 +   0×0)  = 1
// Rotated: [0, 1].add([5, 5]) = [5.0, 6.0]
```

### Multiple rotations

Composing two rotations of the same direction produces a rotation by the sum of their angles.

```swift
// Rotate 45° twice = 90° total
let rotate45 = [
    [0.707, -0.707],
    [0.707,  0.707]
]

// Compose two 45° rotations
let rotate90 = rotate45.multiplyMatrix(rotate45)

// Verify
[1.0, 0.0].transformedBy(rotate90)
// ≈ [0, 1] (within floating-point precision)
```

### Combining different transformations

Scaling, shearing, and rotation can be composed into a single matrix that captures all three operations. The rightmost matrix is applied first.

```swift
// Complex transformation: scale, shear, then rotate
let scale = [Double].diag([2.0, 1.5])

// Horizontal shear with factor 0.3
let shear = [
    [1.0, 0.3],
    [0.0, 1.0]
]

// 90° counterclockwise rotation
let rotate = [
    [0.0, -1.0],
    [1.0,  0.0]
]

// Compose (rightmost applied first)
let complex = rotate.multiplyMatrix(shear).multiplyMatrix(scale)

// Apply to many vectors efficiently
let vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
let transformed = vectors.map { $0.transformedBy(complex) }
```

## Transformation pipelines

Graphics applications commonly use transformation pipelines that move a vertex through several coordinate spaces — object to world, world to camera, camera to screen. Each space is reached by composing a matrix onto the previous one.

```swift
// Object → World → Camera → Screen
let objectToWorld = objectTransform
let worldToCamera = cameraTransform
let cameraToScreen = projectionTransform

// Compose entire pipeline
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
// Rotate 90°, then rotate -90° returns to original
let rotate = [
    [0.0, -1.0],
    [1.0,  0.0]
]

// -90° rotation (clockwise)
let unrotate = [
    [ 0.0, 1.0],
    [-1.0, 0.0]
]

let identity = rotate.multiplyMatrix(unrotate)
// ≈ [[1,0],[0,1]] (within floating-point precision)

// Scale by 2, then scale by 1/2 returns to original
let scale = [Double].diag([2.0, 2.0])
let unscale = [Double].diag([0.5, 0.5])

let identityFromScale = scale.multiplyMatrix(unscale)
// [[1,0],[0,1]]
```

## Performance optimization

### Precompute complex transformations

When the same chain of transformations is applied to many vectors, composing the matrices once and applying the result repeatedly is significantly faster than applying each matrix individually to every vector.

```swift
// Recomputes the chain for every vertex
for vertex in vertices {
    let transformed = vertex
        .transformedBy(scale)
        .transformedBy(rotate)
        .transformedBy(shear)
}

// Compose once, apply many times
let combined = shear.multiplyMatrix(rotate).multiplyMatrix(scale)
for vertex in vertices {
    let transformed = vertex.transformedBy(combined)
}
```

The first pattern recomputes the composed transformation for every vertex; the second precomputes it once and reuses the cached matrix.

### Matrix multiplication complexity

Matrix multiplication is `O(n³)` for n×n matrices. Matrix-vector multiplication is `O(n²)`. For 2D transformations, that means a matrix-matrix multiply costs 8 scalar multiplications and 4 additions, while a matrix-vector multiply costs 4 multiplications and 2 additions.

The implication for scenes with many vertices is direct. Composing the chain once and applying it to many vectors trades a single `O(n³)` operation for many cheaper `O(n²)` operations.

```swift
// Transform 1000 vertices
let vertices: [[Double]] = // 1000 2D points

// Inefficient: 1000 × (2 matrix-vector) = 12,000 operations
let result1 = vertices.map { $0.transformedBy(A).transformedBy(B) }

// Efficient: 1 matrix-matrix + 1000 matrix-vector = 6,012 operations
let AB = A.multiplyMatrix(B)
let result2 = vertices.map { $0.transformedBy(AB) }
```

## See also

- <doc:Matrix-Transformations>
- <doc:Common-Transformations>
- <doc:Vector-Operations>

## Topics

### Transformation operations
- ``Swift/Array/transformedBy(_:)``

### Matrix creation
- ``Swift/Array/identity(_:)``
- ``Swift/Array/diag(_:)``
