# Shape and Size

Inspect dimensions and reshape between vectors and matrices.

## Overview

Every matrix has a shape, a row count and a column count, that defines its structure. Knowing a matrix's shape is essential before performing operations like addition, multiplication, or reshaping, because these operations impose constraints on how dimensions must align. Quiver provides two computed properties that make dimensional inspection straightforward: `.shape` for the row-column structure and `.size` for the total element count. When the shape is wrong for the next operation, `.reshaped(rows:columns:)` and `.flattened()` convert between one-dimensional vectors and two-dimensional matrices without altering the underlying data.

> Note: For a conceptual introduction to matrices and their representations, see [Matrices](https://waynewbishop.github.io/swift-algorithms/21-matrices.html) in Swift Algorithms & Data Structures.

### One-dimensional arrays

For one-dimensional vectors like `[Double]`, Swift's built-in `.count` property already returns the total number of elements. The type system makes the shape obvious: a `[Double]` with `.count == 5` is unambiguously a 5-element vector. There is no row-column distinction to inspect so `.count` is all we need:

```swift
let vector: [Double] = [1.0, 2.0, 3.0, 4.0, 5.0]
vector.count  // 5 — the shape and size in one property
```

In many numerical computing environments, a separate property is needed to query whether an array is one-dimensional, two-dimensional, or higher. In Swift, this information is already encoded in the type signature. The compiler distinguishes between `[Double]` (a vector) and `[[Double]]` (a matrix): the concept of "number of dimensions" is resolved at compile time rather than queried at runtime.

### Inspecting matrix shape

Use `.shape` to retrieve a named tuple describing the matrix dimensions:

```swift
import Quiver

let matrix: [[Double]] = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]

let dimensions = matrix.shape
// dimensions.rows == 2
// dimensions.columns == 3
```

The `.shape` property returns a named tuple of type `(rows: Int, columns: Int)`. The labels `.rows` and `.columns` are built into the return type, so they are available automatically; we do not need to declare them ourselves. This makes intent explicit at the call site and eliminates any chance of transposing the two values by accident.

### Counting total elements

Use `.size` to get the total number of elements across all dimensions:

```swift
import Quiver

let data: [[Double]] = [
    [10.0, 20.0, 30.0],
    [40.0, 50.0, 60.0]
]

data.size       // 6
data.count      // 2 (number of rows only)
```

> Warning: On a two-dimensional array, `.count` returns the number of **rows**, not the total number of elements. This is because Quiver represents matrices as nested Swift arrays. To get the true element count across all rows and columns, use `.size` instead.

### Reshaping vectors to matrices

Use `.reshaped(rows:columns:)` to convert a one-dimensional array into a two-dimensional matrix. Elements fill the matrix row by row (row-major order):

```swift
import Quiver

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

let flat = matrix.flattened()
// [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
```

Flattening and reshaping are complementary. Flattening a matrix and reshaping back to the original dimensions returns the same matrix:

```swift
let original = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
let restored = original.flattened().reshaped(rows: 2, columns: 3)
// [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
```

### Changing matrix dimensions

Reshape a matrix directly to different dimensions. The matrix is flattened internally, then filled into the new shape:

```swift
import Quiver

let wide = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

let tall: [[Double]] = wide.reshaped(rows: 3, columns: 2)
// [[1.0, 2.0],
//  [3.0, 4.0],
//  [5.0, 6.0]]

let row: [[Double]] = wide.reshaped(rows: 1, columns: 6)
// [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]

let column: [[Double]] = wide.reshaped(rows: 6, columns: 1)
// [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
```

### Practical applications

**Validate dimensions before operations:**

```swift
import Quiver

let a: [[Double]] = [[1.0, 2.0], [3.0, 4.0]]
let b: [[Double]] = [[5.0, 6.0], [7.0, 8.0]]

guard a.shape == b.shape else {
    fatalError("Dimension mismatch")
}
let sum = a.add(b)
```

**Inspect dimensions before processing:**

```swift
import Quiver

// Weekly sales data: 4 stores × 7 days
let sales: [[Double]] = [
    [120, 95, 110, 130, 140, 200, 180],
    [85, 70, 90, 75, 95, 150, 130],
    [200, 180, 210, 190, 220, 300, 280],
    [60, 55, 70, 65, 80, 110, 95]
]

let (stores, days) = sales.shape
// stores == 4, days == 7

sales.size  // 28 (total data points across all stores)
```

Here we use tuple destructuring to unpack `.shape` into names that reflect the domain: `stores` and `days` instead of the generic `rows` and `columns`. Destructuring works by position, so the first value is always the row count and the second is always the column count.

**Prepare data for matrix operations:**

```swift
let readings = [23.1, 45.2, 67.3, 89.4, 12.5, 34.6]

let samples = readings.reshaped(rows: 3, columns: 2)
// [[23.1, 45.2],   — Sample 1: temperature, humidity
//  [67.3, 89.4],   — Sample 2
//  [12.5, 34.6]]   — Sample 3
```

**Convert between row and column vectors:**

```swift
let v = [1.0, 2.0, 3.0]

let columnVec = v.reshaped(rows: 3, columns: 1)
let rowVec = v.reshaped(rows: 1, columns: 3)
```

### Preconditions

- **Reshape**: Total element count must equal `rows × columns`
- **Reshape**: Dimensions must be positive (greater than zero)
- **Flatten**: Empty matrices return an empty array

### Compile-time type guarantees

Both `.shape` and `.size` are constrained to two-dimensional arrays whose inner elements conform to `Numeric`. The compiler enforces this at build time — attempting to call `.shape` on a `[String]` or a flat `[Double]` produces a compile-time error, not a runtime crash. Invalid dimensional queries are caught before the code ever runs, eliminating an entire class of bugs that would otherwise surface only during execution.

This is a broader benefit of working in a strongly typed, compiled language. Properties that other environments require for runtime type inspection are unnecessary here. When we declare `[[Double]]`, the element type is fixed and enforced by the compiler. There is no silent type coercion, no unexpected precision loss, and no need to inspect the array to discover what it contains. The type signature is the single source of truth.

> Experiment: **The Quiver Notebook** is the right place to validate dimensions before a model ever touches the data. Load a bundled dataset, pull a feature matrix out with `toMatrix(columns:)`, and print `.shape` and `.size` before fitting. Then deliberately mismatch, passing a feature matrix and a label vector whose row count does not match, and watch what happens. A `guard features.shape.rows == labels.count else { ... }` check at the top of a snippet catches the kind of bug that produces wrong predictions silently. See <doc:Quiver-Notebook>.
