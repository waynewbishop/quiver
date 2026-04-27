# Matrix Operations

Work with two-dimensional arrays (matrices) using element-wise arithmetic and transformations.

## Overview

A **matrix** is a rectangular grid of numbers arranged in rows and columns. While a `vector` is a single list of numbers representing a point or direction in space, a matrix is a collection of multiple vectors organized together. Quiver extends Swift's `Array` type to support matrix operations on nested arrays. Matrices represent tabular data, transformations, and multi-dimensional datasets. For the mathematical foundations, see <doc:Linear-Algebra-Primer>. 

> Tip: For detailed coverage of matrix concepts with visualizations and examples, see [Matrices](https://waynewbishop.github.io/swift-algorithms/21-matrices.html) in Swift Algorithms & Data Structures.

### Creating matrices

```swift
import Quiver

// Simple 2×3 matrix
let matrix = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]

// Inspect dimensions
matrix.shape  // (rows: 2, columns: 3)
matrix.size   // 6

// Access elements
let value = matrix[0][1]  // 2.0 (row 0, column 1)

// Access rows
let firstRow = matrix[0]  // [1.0, 2.0, 3.0]

// Access columns (using Quiver)
let secondColumn = matrix.column(at: 1)  // [2.0, 5.0]
```

### Element-wise arithmetic

Matrix arithmetic operations work element-by-element:

```swift
let m1 = [[1.0, 2.0], [3.0, 4.0]]
let m2 = [[5.0, 6.0], [7.0, 8.0]]

// Perform element-wise arithmetic on two matrices
let sum = m1.add(m2)            // [[6.0, 8.0], [10.0, 12.0]]
let difference = m1.subtract(m2)  // [[-4.0, -4.0], [-4.0, -4.0]]
let product = m1.multiply(m2)     // [[5.0, 12.0], [21.0, 32.0]] (Hadamard)
let quotient = m1.divide(m2)      // [[0.2, 0.33...], [0.42..., 0.5]]
```

> Important: The `multiply(_:)` method performs **element-wise** multiplication (Hadamard product), not matrix multiplication. For matrix multiplication, use `.multiplyMatrix()`. See <doc:Matrix-Transformations> for how matrix-vector multiplication uses the dot product internally.

### Scalar broadcasting

Apply a scalar value to every element:

```swift
let matrix = [[100.0, 200.0], [300.0, 400.0]]

// Apply a scalar operation to every element
let scaled = matrix * 0.5        // [[50.0, 100.0], [150.0, 200.0]]
let shifted = matrix + 10.0      // [[110.0, 210.0], [310.0, 410.0]]
let divided = matrix / 100.0     // [[1.0, 2.0], [3.0, 4.0]]

// Commutative operations work in either direction
let doubled = 2.0 * matrix       // Same as matrix * 2.0
let offset = 5.0 + matrix        // Same as matrix + 5.0
```

### Common patterns

**Data standardization (z-score):**
```swift
// Normalize each element to zero mean and unit variance
let data = [[100.0, 200.0], [300.0, 400.0]]
let mean = 250.0
let stdDev = 129.0
let standardized = (data - mean) / stdDev
```

**Feature scaling (min-max):**
```swift
// Scale features to the 0-1 range
let features = [[1.0, 5.0], [3.0, 7.0]]
let min = 1.0
let max = 7.0
let scaled = (features - min) / (max - min)
```

**Applying transformations:**
```swift
// Calibrate measurements with a 5% increase and 2.0 offset
let measurements = [[10.0, 20.0], [30.0, 40.0]]
let calibrated = measurements * 1.05 + 2.0
```

### Transpose

Flip rows and columns:

```swift
let matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let transposed = matrix.transpose()
// Result: [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
```

### Matrix multiplication

For matrix multiplication, use `.multiplyMatrix()`:

```swift
let a = [[1.0, 2.0], [3.0, 4.0]]
let b = [[5.0, 6.0], [7.0, 8.0]]
let product = a.multiplyMatrix(b)
// Result: [[19.0, 22.0], [43.0, 50.0]]
```

### Determinant

Calculate the determinant of a square matrix:

```swift
let matrix = [[4.0, 3.0], [6.0, 3.0]]
let det = matrix.determinant  // -6.0
```

The [determinant](<doc:Determinants-Primer>) provides important information about a matrix:
- `det = 0`: Matrix is singular (not invertible)
- `det ≠ 0`: Matrix is invertible
- Magnitude indicates volume scaling in geometric transformations

### Matrix inversion

Compute the inverse of a matrix:

```swift
let matrix = [[4.0, 7.0], [2.0, 6.0]]
let inverse = try matrix.inverted()
// [[0.6, -0.7], [-0.2, 0.4]]

// Verify: A × A⁻¹ = I (identity)
let identity = matrix.multiplyMatrix(inverse)
// [[1.0, 0.0], [0.0, 1.0]]
```

The inverse matrix A⁻¹ satisfies: A × A⁻¹ = I (identity matrix)

**Common uses:**
- Solving linear systems: Ax = b becomes x = A⁻¹b
- Reversing transformations
- Computing least squares solutions

For educational clarity, convert results to fractional form with `asFractions`:

```swift
let inverse = try matrix.inverted()
inverse.asFractions()
// [[3/5, -7/10], [-1/5, 2/5]]
```

This reveals the rational structure behind decimal results. See <doc:Determinants-Primer> for a deeper look at why inverse matrices produce clean fractions.

> Important: Only non-singular matrices (determinant ≠ 0) can be inverted. Calling `.inverted()` on a singular matrix throws `MatrixError.singular`.

### Working with data

Matrices naturally represent tabular data:

```swift
// Game scores: rows = players, columns = games
let scores = [
    [95.0, 88.0, 92.0],  // Player A
    [87.0, 90.0, 89.0],  // Player B
    [92.0, 94.0, 88.0]   // Player C
]

scores.shape  // (rows: 3, columns: 3) — 3 players, 3 games

// Extract all scores from game 2
let game2Scores = scores.column(at: 1)  // [88.0, 90.0, 94.0]

// Calculate average for Player B
let playerB = scores[1]
let average = playerB.mean() ?? 0.0  // 88.67
```

### Document-term matrices

Matrices organize text data for analysis:

```swift
// Document-term matrix: rows = documents, columns = words
let documents = [
    [2.0, 3.0, 1.0],  // Doc 1: word counts
    [1.0, 2.0, 3.0],  // Doc 2: word counts
    [3.0, 1.0, 2.0]   // Doc 3: word counts
]

// Analyze how often word 1 appears across documents
let word1Usage = documents.column(at: 0)  // [2.0, 1.0, 3.0]

// Switch to term-document orientation
let byTerms = documents.transpose()

// Collapse into a single vector for total word frequency
let allCounts = documents.flattened()  // [2.0, 3.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0, 2.0]
```

### Type support

Matrix operations work with both `Double` and `Float` types:

```swift
let doubleMatrix: [[Double]] = [[1.0, 2.0], [3.0, 4.0]]
let floatMatrix: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]

// Both support full operator syntax
let result1 = doubleMatrix * 2.0
let result2 = floatMatrix * 2.0
```

### Preconditions

- **Element-wise operations**: Matrices must have same dimensions (rows × columns)
- **Matrix multiplication**: Inner dimensions must match (n×k) × (k×m)
- **Division**: Divisor cannot contain zero elements
- **Column access**: Index must be within column count

