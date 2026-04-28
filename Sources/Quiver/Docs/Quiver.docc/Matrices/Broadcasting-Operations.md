# Broadcasting Operations

Apply operations between arrays and scalars or between arrays of different dimensions.

## Overview

Quiver provides **broadcasting** capabilities that enable operations between arrays and scalars, or between arrays of different shapes. Broadcasting produces cleaner, more expressive code by eliminating explicit loops for common element-wise operations. 

### Why broadcasting?

Swift already provides powerful functional methods like `map`, `reduce`, and `filter` for transforming arrays. Broadcasting is a standard approach for mathematical operations on arrays, providing a more declarative syntax specifically designed for numerical computing. While Swift's functional methods are excellent for general-purpose transformations, broadcasting reads more like mathematical notation.

**Swift's map (functional approach):**
```swift
import Quiver

let temperatures = [72.0, 68.0, 73.0, 70.0, 75.0]

// Convert Fahrenheit to Celsius: (F - 32) * 5/9
let celsiusMap = temperatures.map { ($0 - 32.0) * 5.0/9.0 }
// [22.2, 20.0, 22.8, 21.1, 23.9]
```

**Quiver's broadcasting (mathematical approach):**
```swift
// Subtract and multiply using operator broadcasting
let celsius = (temperatures - 32.0) * 5.0/9.0
// [22.2, 20.0, 22.8, 21.1, 23.9]

// Method chaining also available
let celsiusAlt = temperatures.broadcast(subtracting: 32.0)
                            .broadcast(multiplyingBy: 5.0/9.0)
```

**When to use broadcasting vs map:**

Broadcasting is ideal for applying scalar mathematical operations to arrays. It makes the code read like mathematical notation and clearly separates each transformation step. Use `map` for custom logic, complex transformations, or non-mathematical operations where broadcasting doesn't apply.

```swift
let temperatures = [72.0, 68.0, 73.0, 70.0, 75.0]
let scores = [88.0, 91.0, 76.0, 95.0, 82.0]

// Broadcasting excels at scalar mathematical operations
let normalized = (scores - 79.0) / 8.0
let scaled = temperatures * 2.0

// Map excels at custom or non-numeric transformations
let formatted = temperatures.map { "\($0)°F" }
let categorized = scores.map { $0 >= 90 ? "A" : "B" }
```

### Scalar broadcasting

The simplest form of broadcasting applies a scalar operation to every element in an array:

```swift
let vector = [1.0, 2.0, 3.0, 4.0]

// Add a scalar to each element
let increased = vector.broadcast(adding: 5.0)  // [6.0, 7.0, 8.0, 9.0]

// Multiply each element by a scalar
let scaled = vector.broadcast(multiplyingBy: 2.0)  // [2.0, 4.0, 6.0, 8.0]

// Subtract a scalar from each element
let decreased = vector.broadcast(subtracting: 1.0)  // [0.0, 1.0, 2.0, 3.0]

// Divide each element by a scalar (floating-point only)
let divided = vector.broadcast(dividingBy: 2.0)  // [0.5, 1.0, 1.5, 2.0]
```

> Tip: Broadcasting operations create new arrays without modifying the original array, maintaining Swift's value semantics.

### Operator-based broadcasting

Scalar broadcasting is also available through standard arithmetic operators:

**Vector broadcasting:**
```swift
let vector = [1.0, 2.0, 3.0]

// Add a scalar using method syntax
let result1 = vector.broadcast(adding: 10.0)

// Add a scalar using operator syntax
let result2 = vector + 10.0
let result3 = vector * 2.0
let result4 = vector / 3.0

// Commutative operations work in either direction
let result5 = 10.0 + vector
let result6 = 2.0 * vector
```

**Matrix broadcasting:**
```swift
let matrix = [[1.0, 2.0], [3.0, 4.0]]

// Multiply each element using method syntax
let result1 = matrix.map { $0.broadcast(multiplyingBy: 2.0) }

// Multiply each element using operator syntax
let result2 = matrix * 2.0
let result3 = matrix + 10.0
let result4 = (matrix - 5.0) / 2.0
```

The operator syntax is recommended for new code as it improves readability. The method-based syntax remains available for compatibility and for cases requiring custom operations via closures.

### Array-to-array operations

Operators (`+`, `-`, `*`, `/`) cover the scalar-on-array case. Element-wise operations between two arrays of the same shape use named methods instead — `add`, `subtract`, `multiply`, and `divide`:

```swift
let a = [1.0, 2.0, 3.0]
let b = [10.0, 20.0, 30.0]

let sum  = a.add(b)        // [11.0, 22.0, 33.0]
let diff = a.subtract(b)   // [-9.0, -18.0, -27.0]
let prod = a.multiply(b)   // [10.0, 40.0, 90.0]
let quot = a.divide(b)     // [0.1, 0.1, 0.1]
```

The same methods work between two matrices of the same shape:

```swift
let m1 = [[1.0, 2.0], [3.0, 4.0]]
let m2 = [[10.0, 20.0], [30.0, 40.0]]

let summed = m1.add(m2)         // [[11.0, 22.0], [33.0, 44.0]]
let scaled = m1.multiply(m2)    // [[10.0, 40.0], [90.0, 160.0]]
```

> Tip: Operators are intentionally reserved for scalar-on-array broadcasting. Quiver does not overload `+`, `-`, `*`, or `/` between two arrays to keep the standard library's array semantics intact and to make the intent of every expression explicit at the call site.

### Matrix-vector broadcasting

Broadcasting also allows operations between matrices (2D arrays) and vectors:

```swift
let matrix = [[1.0, 2.0, 3.0], 
              [4.0, 5.0, 6.0]]

let rowVector = [10.0, 20.0, 30.0]
let columnVector = [100.0, 200.0]

// Add a row vector to each row
let rowBroadcast = matrix.broadcast(addingToEachRow: rowVector)
// [[11.0, 22.0, 33.0], 
//  [14.0, 25.0, 36.0]]

// Add a column vector to each column
let columnBroadcast = matrix.broadcast(addingToEachColumn: columnVector)
// [[101.0, 102.0, 103.0], 
//  [204.0, 205.0, 206.0]]

// Multiply each row by a vector
let rowMultiply = matrix.broadcast(multiplyingEachRowBy: rowVector)
// [[10.0, 40.0, 90.0], 
//  [40.0, 100.0, 180.0]]

// Multiply each column by a vector
let columnMultiply = matrix.broadcast(multiplyingEachColumnBy: columnVector)
// [[100.0, 200.0, 300.0], 
//  [800.0, 1000.0, 1200.0]]
```

> Important: When broadcasting vectors across matrices, the dimensions must be compatible. Row vectors must have the same length as matrix columns, and column vectors must have the same length as matrix rows.

### Custom broadcasting operations

For more flexibility, Quiver provides custom broadcasting operations with closures:

```swift
let vector = [1, 2, 3, 4]

// Apply a custom operation with a scalar (raising each element to the power of 2)
let powered = vector.broadcast(with: 2) { element, exponent in
    Int(pow(Double(element), Double(exponent)))
}  // [1, 4, 9, 16]

// Multiply each element by 3
let tripled = vector.broadcast(with: 3) { element, multiplier in
    element * multiplier
} // [3, 6, 9, 12]

// Custom matrix-vector operations
let matrix = [[1, 2, 3], [4, 5, 6]]
let rowVector = [10, 100, 1000]

// Apply a custom operation between rows and a vector
let customRowOperation = matrix.broadcast(withRowVector: rowVector) { matrixElement, vectorElement in
    matrixElement * vectorElement + matrixElement  // matrixElement * (vectorElement + 1)
}
// [[11, 202, 3003], 
//  [44, 505, 6006]]
```

> Important: In the closure, the first parameter always represents the element from the array/matrix, and the second parameter represents the scalar value or the corresponding element from the broadcast vector.

> Tip: Choose descriptive parameter names in closures that reflect the specific operation being performed, rather than using generic names like "a" and "b".

### Use cases

Broadcasting operations are particularly useful for:

- **Data standardization**: Subtract means or divide by standard deviations (z-score)
- **Feature scaling**: Apply weights to feature vectors
- **Signal processing**: Apply filters or transformations to signals
- **Image processing**: Adjust color channels or apply transformations
- **Financial calculations**: Apply interest rates or time factors

### Implementation details

Quiver implements broadcasting through extension methods on `Array` with appropriate type constraints:

- Scalar broadcasting is available for arrays with `Numeric` elements
- Division operations are only available for arrays with `FloatingPoint` elements
- Matrix-vector broadcasting requires collections of `Numeric` elements

The broadcasting implementations verify dimension compatibility at runtime, providing clear error messages when dimensions don't match.

## Topics

### Scalar broadcasting
- ``Swift/Array/broadcast(adding:)``
- ``Swift/Array/broadcast(multiplyingBy:)``
- ``Swift/Array/broadcast(subtracting:)``
- ``Swift/Array/broadcast(dividingBy:)``

### Matrix-vector broadcasting
- ``Swift/Array/broadcast(addingToEachRow:)``
- ``Swift/Array/broadcast(addingToEachColumn:)``
- ``Swift/Array/broadcast(multiplyingEachRowBy:)``
- ``Swift/Array/broadcast(multiplyingEachColumnBy:)``

### Custom broadcasting
- ``Swift/Array/broadcast(with:operation:)``
- ``Swift/Array/broadcast(withRowVector:operation:)``
- ``Swift/Array/broadcast(withColumnVector:operation:)``

### Related articles
- <doc:Matrix-Operations>
