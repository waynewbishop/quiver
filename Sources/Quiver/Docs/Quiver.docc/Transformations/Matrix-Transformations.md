# Matrix Transformations

Apply transformations to vectors using matrices.

## Overview

Matrices transform vectors through multiplication. A transformation matrix modifies a vector's position, orientation, or scale in space. Common transformations include rotation, scaling, reflection, and shearing. Understanding how matrices act on vectors is the foundation for graphics programming, physics simulations, and machine learning pipelines.

> Tip: For geometric intuition about how transformations work, including basis vectors, coordinate systems, and visual examples, see [Matrix Transformations](https://waynewbishop.github.io/swift-algorithms/22-matrix-transformations.html) in Swift Algorithms & Data Structures.

## Basic usage

Transform a vector by multiplying it with a transformation matrix:

```swift
import Quiver

// A 90° counterclockwise rotation matrix
let rotation = [
    [0.0, -1.0],
    [1.0,  0.0]
]

let vector = [1.0, 0.0]
let rotated = vector.transformedBy(rotation)
// [0.0, 1.0] - vector now points up
```

## Matrix-vector multiplication

Matrix-vector multiplication uses the **dot product** between each matrix row and the vector. Each row of the matrix produces one element of the result:

```swift
// Scaling matrix doubles x and triples y
let matrix = [
    [2.0, 0.0],
    [0.0, 3.0]
]

let v = [4.0, 5.0]
let result = v.transformedBy(matrix)
// [8.0, 15.0]
```

**How it works:**
```
Row 1: [2, 0] • [4, 5] = (2×4 + 0×5) = 8
Row 2: [0, 3] • [4, 5] = (0×4 + 3×5) = 15
Result: [8, 15]
```

> Important: The matrix must have the same number of **columns** as the vector has **elements**. A 2×2 matrix transforms 2D vectors; a 3×3 matrix transforms 3D vectors.

**Quiver provides two equivalent syntaxes:**

```swift
// Vector perspective (recommended)
let transformed = vector.transformedBy(matrix)

// Matrix perspective
let transformed2 = matrix.transform(vector)
```

## Basis vectors and coordinate systems

Every 2D transformation matrix is defined by where it sends the two basis vectors — `i-hat` [1, 0] and `j-hat` [0, 1]. The first column of the matrix is where i-hat lands; the second column is where j-hat lands:

```swift
// Identity: basis vectors stay in place
let identity = [Double].identity(2)
// Column 1: [1, 0] — i-hat stays at [1, 0]
// Column 2: [0, 1] — j-hat stays at [0, 1]

// Rotation: basis vectors rotate together
let rotate90 = [
    [0.0, -1.0],
    [1.0,  0.0]
]
// Column 1: [0, 1] — i-hat moves to [0, 1]
// Column 2: [-1, 0] — j-hat moves to [-1, 0]
```

This column-based perspective explains why matrix-vector multiplication works: the result is a linear combination of the matrix columns, weighted by the vector components. For a vector [a, b], the transformed result equals `a × column1 + b × column2`.

```swift
// Transform [3, 2] using scaling matrix
let scale = [[2.0, 0.0],
             [0.0, 3.0]]

// Equivalent: 3 × [2, 0] + 2 × [0, 3] = [6, 0] + [0, 6] = [6, 6]
let result = [3.0, 2.0].transformedBy(scale)
// [6.0, 6.0]
```

## Creating transformation matrices

### Identity matrix

The identity matrix leaves vectors unchanged — it maps every vector to itself:

```swift
// Create a 2×2 identity matrix
let identity = [Double].identity(2)
// [[1.0, 0.0],
//  [0.0, 1.0]]

[3.0, 4.0].transformedBy(identity)  // [3.0, 4.0]
```

The identity matrix serves as the starting point for building transformations and is the neutral element in matrix multiplication: `A × I = I × A = A`.

### Diagonal matrices

Diagonal matrices have values only on the main diagonal and zeros elsewhere. They scale each axis independently:

```swift
// Create diagonal matrix from a vector of scale factors
let scale = [Double].diag([2.0, 3.0])
// [[2.0, 0.0],
//  [0.0, 3.0]]

[4.0, 5.0].transformedBy(scale)  // [8.0, 15.0]
```

When all diagonal values are equal, the matrix performs uniform scaling. When they differ, it stretches or compresses along individual axes.

## Common transformations

For detailed transformation examples (rotation, scaling, reflection, shearing), see <doc:Common-Transformations>.

### Rotation example

```swift
// 45° counterclockwise rotation
let rotate45 = [
    [0.707, -0.707],
    [0.707,  0.707]
]

let v = [1.0, 0.0]
let rotated = v.transformedBy(rotate45)
// Row 1: [0.707, -0.707] • [1, 0] = (0.707×1 + (-0.707)×0) = 0.707
// Row 2: [0.707,  0.707] • [1, 0] = (0.707×1 +   0.707×0)  = 0.707
// Result: [0.707, 0.707]
```

Rotation matrices preserve vector magnitude — the length of the vector stays the same before and after rotation. This property makes rotation matrices **orthogonal**: their inverse equals their transpose.

### Scaling example

```swift
// Uniform scaling — same factor in all directions
let scale2x = [Double].diag([2.0, 2.0])
[3.0, 4.0].transformedBy(scale2x)  // [6.0, 8.0]

// Non-uniform scaling — different factors per axis
let stretch = [Double].diag([3.0, 0.5])
[2.0, 4.0].transformedBy(stretch)  // [6.0, 2.0]
```

## Topics

### Matrix creation
- ``Swift/Array/identity(_:)``
- ``Swift/Array/diag(_:)``

### Transformation operations
- ``Swift/Array/transformedBy(_:)``
