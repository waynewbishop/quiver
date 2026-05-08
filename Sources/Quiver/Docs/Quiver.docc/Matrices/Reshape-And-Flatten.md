# Reshape and Flatten

Convert between one-dimensional vectors and two-dimensional matrices.

## Overview

Reshaping changes the dimensions of an array without altering its data. A 1D vector of 12 elements can become a 3×4 matrix, a 2×6 matrix, or any other shape where the total element count stays the same. Flattening is the reverse — collapsing a matrix back into a single vector. These operations are fundamental to data preparation, where raw data often needs to be reorganized into the shape that an algorithm expects.

> Note: For a conceptual introduction to matrices and their representations, see [Matrices](https://waynewbishop.github.io/swift-algorithms/21-matrices.html) in Swift Algorithms & Data Structures.

### Reshaping vectors to matrices

Use `.reshaped(rows:columns:)` to convert a one-dimensional array into a two-dimensional matrix. Elements fill the matrix row by row (row-major order):

```swift
import Quiver

// Generate a sequence and reshape into a matrix
let values = [Double].arange(1, 13)  // [1, 2, 3, ..., 12]
let matrix = values.reshaped(rows: 3, columns: 4)
// [[1.0, 2.0, 3.0, 4.0],
//  [5.0, 6.0, 7.0, 8.0],
//  [9.0, 10.0, 11.0, 12.0]]
```

The total number of elements must equal `rows × columns`. Attempting to reshape 12 elements into a 3×5 matrix triggers a precondition failure since 3 × 5 = 15 ≠ 12.

### Flattening matrices to vectors

Use `.flattened()` to collapse a matrix into a one-dimensional array by concatenating rows:

```swift
import Quiver

let matrix = [
    [10.0, 20.0, 30.0],
    [40.0, 50.0, 60.0]
]

// Flatten to a single vector
let flat = matrix.flattened()
// [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
```

Flattening and reshaping are complementary operations. Flattening a matrix and reshaping back to the original dimensions returns the same matrix:

```swift
let original = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let restored = original.flattened().reshaped(rows: 2, columns: 3)
// [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
```

### Changing matrix dimensions

Reshape a matrix directly to different dimensions. The matrix is flattened internally, then filled into the new shape:

```swift
import Quiver

// Start with a 2×3 matrix
let wide = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

// Reshape to 3×2
let tall: [[Double]] = wide.reshaped(rows: 3, columns: 2)
// [[1.0, 2.0],
//  [3.0, 4.0],
//  [5.0, 6.0]]

// Reshape to single row
let row: [[Double]] = wide.reshaped(rows: 1, columns: 6)
// [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]

// Reshape to single column
let column: [[Double]] = wide.reshaped(rows: 6, columns: 1)
// [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
```

### Common patterns

**Prepare data for matrix operations:**
```swift
// Raw sensor readings arrive as a flat stream
let readings = [23.1, 45.2, 67.3, 89.4, 12.5, 34.6]

// Reshape into samples × features for analysis
let samples = readings.reshaped(rows: 3, columns: 2)
// [[23.1, 45.2],   — Sample 1: temperature, humidity
//  [67.3, 89.4],   — Sample 2
//  [12.5, 34.6]]   — Sample 3
```

**Convert between row and column vectors:**
```swift
let v = [1.0, 2.0, 3.0]

// Column vector (3×1 matrix)
let columnVec = v.reshaped(rows: 3, columns: 1)

// Row vector (1×3 matrix)
let rowVec = v.reshaped(rows: 1, columns: 3)
```

### Preconditions

- **Reshape**: Total element count must equal `rows × columns`
- **Reshape**: Dimensions must be positive (greater than zero)
- **Flatten**: Empty matrices return an empty array

