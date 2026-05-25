# Matrix Transformations

Apply transformations to vectors using matrices.

## Overview

Matrices transform vectors through multiplication. A transformation matrix modifies a vector's position, orientation, or scale in space. Common transformations include rotation, scaling, reflection, and shearing. Understanding how matrices act on vectors is the foundation for graphics programming, physics simulations, and machine learning pipelines.

> Note: For geometric intuition about how transformations work, including basis vectors, coordinate systems, and visual examples, see [Matrix Transformations](https://waynewbishop.github.io/swift-algorithms/22-matrix-transformations.html) in Swift Algorithms & Data Structures.

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

## Rotation

Rotation transforms rotate vectors around the origin by a specified angle. In 2D, positive angles produce counterclockwise rotation.

### Common rotation matrices

A rotation by 90° counterclockwise sends the unit vector along the x-axis to the unit vector along the y-axis, and the unit vector along the y-axis to the negative x-axis.

```swift
import Quiver

// Rotate vectors 90° counterclockwise around the origin
let rotate90 = [
    [0.0, -1.0],
    [1.0,  0.0]
]

// Basis vectors move: i-hat [1,0] → [0,1], j-hat [0,1] → [-1,0]

[1.0, 0.0].transformedBy(rotate90)
// Row 1: [0, -1] • [1, 0] = (0×1 + (-1)×0) = 0
// Row 2: [1,  0] • [1, 0] = (1×1 +   0×0)  = 1
// Result: [0.0, 1.0]
```

A 180° rotation reverses every vector by negating both components, while a 45° rotation produces the canonical √2/2 ≈ 0.707 entries.

```swift
// Rotate vectors 180° to reverse direction
let rotate180 = [
    [-1.0,  0.0],
    [ 0.0, -1.0]
]

// Rotate vectors 45° counterclockwise
let rotate45 = [
    [0.707, -0.707],
    [0.707,  0.707]
]

[3.0, 4.0].transformedBy(rotate180)  // [-3.0, -4.0]
[1.0, 0.0].transformedBy(rotate45)   // [0.707, 0.707] — 45° between x and y axes
```

A clockwise rotation is the transpose of its counterclockwise counterpart, which mirrors the off-diagonal signs.

```swift
// Rotate vectors 90° clockwise
let rotate90cw = [
    [ 0.0, 1.0],
    [-1.0, 0.0]
]

[1.0, 0.0].transformedBy(rotate90cw)  // [0.0, -1.0] — vector now points down
```

Rotation matrices preserve vector magnitude — the length of the vector stays the same before and after rotation. This property makes rotation matrices **orthogonal**: their inverse equals their transpose.

### Practical examples

Rotation matrices appear throughout interactive graphics — turning a character's facing direction, animating an object along a circular path, orienting a sprite. The same matrix that rotates `[1, 0]` to `[0, 1]` rotates any vector by the same angle.

```swift
// Rotate a character's facing direction
let facingRight = [1.0, 0.0]

let facingUp = facingRight.transformedBy(rotate90)    // [0.0, 1.0]
let facingLeft = facingRight.transformedBy(rotate180) // [-1.0, 0.0]

// Move an object along a circular path around the origin
let radius = 5.0
let position = [radius, 0.0].transformedBy(rotate90)  // [0.0, 5.0]
```

> Experiment: **The Quiver Notebook** is the right place to watch rotation as a continuous transformation. Sweep the angle from 0° through 360° and watch the transformed vector trace the unit circle — at 90° the components swap, at 180° they flip sign. See <doc:Quiver-Notebook>.

## Scaling

Scaling transformations change the magnitude of vectors. Uniform scaling multiplies every dimension by the same factor; non-uniform scaling stretches or compresses individual axes by different amounts.

### Uniform and non-uniform scaling

A diagonal matrix with the same value in every diagonal position scales uniformly. Different diagonal entries produce non-uniform scaling, where each axis stretches or compresses independently.

```swift
// Scale all axes by the same factor
let scale2x = [Double].diag([2.0, 2.0])
// [[2.0, 0.0],
//  [0.0, 2.0]]

[3.0, 4.0].transformedBy(scale2x)
// Row 1: [2, 0] • [3, 4] = (2×3 + 0×4) = 6
// Row 2: [0, 2] • [3, 4] = (0×3 + 2×4) = 8
// Result: [6.0, 8.0]

// Stretch horizontally and compress vertically
let stretch = [
    [3.0, 0.0],
    [0.0, 0.5]
]

[2.0, 4.0].transformedBy(stretch)  // [6.0, 2.0] — 3× wider, half as tall
```

### Practical examples

Scaling underlies sprite resizing, aspect-ratio correction, and camera zoom. A uniform scale preserves shape; a non-uniform scale deliberately distorts it.

```swift
// Double the size of a sprite
let spriteSize = [32.0, 48.0]
let scaled = spriteSize.transformedBy(scale2x)  // [64.0, 96.0]

// Apply a zoom level to camera coordinates
let zoomLevel = 1.5
let zoom = [Double].diag([zoomLevel, zoomLevel])
let cameraCenter = [100.0, 75.0]
let zoomedView = cameraCenter.transformedBy(zoom)  // [150.0, 112.5]
```

For aspect-ratio correction, mapping a 16:9 vector into a square coordinate space requires stretching only the y-axis. The matrix has a 1.0 on the x-axis and `16/9` on the y-axis, leaving the horizontal coordinate unchanged.

## Reflection

Reflection mirrors vectors across an axis or line. The transformation flips coordinates while preserving distances and angles, so a reflected shape has the same size as the original but reversed handedness.

### Reflection matrices

The three most common 2D reflections — across the x-axis, the y-axis, and the diagonal `y = x` — each negate exactly one coordinate or swap them.

```swift
// Mirror a vector across the x-axis (y is negated)
let reflectX = [
    [ 1.0, 0.0],
    [ 0.0, -1.0]
]

// Mirror a vector across the y-axis (x is negated)
let reflectY = [
    [-1.0, 0.0],
    [ 0.0, 1.0]
]

// Swap x and y by reflecting across y=x (this is also the transpose)
let reflectDiagonal = [
    [0.0, 1.0],
    [1.0, 0.0]
]

[3.0, 4.0].transformedBy(reflectX)         // [3.0, -4.0]
[3.0, 4.0].transformedBy(reflectY)         // [-3.0, 4.0]
[3.0, 4.0].transformedBy(reflectDiagonal)  // [4.0, 3.0]
```

### Practical examples

Reflection appears whenever a scene requires a mirror image — a sprite facing the other direction, an object's image below a water line, or a UI element flipped for a right-to-left layout.

```swift
// Flip a sprite to face the opposite direction
let spritePosition = [10.0, 5.0]
let mirrored = spritePosition.transformedBy(reflectY)  // [-10.0, 5.0]

// Reflect an object's position below the water line
let objectPosition = [5.0, 10.0]
let waterReflection = objectPosition.transformedBy(reflectX)  // [5.0, -10.0]
```

## Shear

Shear transformations slant the coordinate system, shifting one axis proportionally to the other. The result is a leaning or skewed effect — rectangles become parallelograms, squares become rhombi.

### Horizontal and vertical shear

A horizontal shear shifts x in proportion to y; a vertical shear shifts y in proportion to x. The shear factor controls how much one axis bleeds into the other.

```swift
// Shift x proportionally to y with a shear factor of 0.5
let shearH = [
    [1.0, 0.5],
    [0.0, 1.0]
]

// Shift y proportionally to x with a shear factor of 0.5
let shearV = [
    [1.0, 0.0],
    [0.5, 1.0]
]

[2.0, 4.0].transformedBy(shearH)  // [4.0, 4.0]
[2.0, 4.0].transformedBy(shearV)  // [2.0, 5.0]
```

### Practical examples

Shear matrices simulate italic text and approximate perspective foreshortening. The italic shear leans letters to the right; the perspective shear shifts horizontal position based on apparent depth.

```swift
// Simulate italic text by leaning letters to the right
let italicShear = [
    [1.0, 0.3],
    [0.0, 1.0]
]

let letterPosition = [10.0, 20.0]
let italicPosition = letterPosition.transformedBy(italicShear)  // [16.0, 20.0]
```

## Topics

### Matrix creation
- ``Swift/Array/identity(_:)``
- ``Swift/Array/diag(_:)``

### Transformation operations
- ``Swift/Array/transformedBy(_:)``
