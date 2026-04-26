# Array Generation

Creating arrays and matrices with specific patterns, fills, and sequences.

## Overview

Most numerical work begins with an array of a known shape and a known fill value — a buffer of zeros to accumulate into, a row of ones to use as a multiplicative identity, an evenly spaced sequence for plotting, or an identity matrix for transformations. Quiver provides a small set of static methods on `Array` for these cases, so the starting array always has the right size and the right values without a manual loop.

### Filling arrays with constants

The most common starting point is an array of a fixed length filled with a single value. `zeros`, `ones`, and `full` cover this case for any numeric type.

```swift
import Quiver

// Create 1D arrays
let zeros = [Double].zeros(5)        // [0.0, 0.0, 0.0, 0.0, 0.0]
let ones = [Int].ones(3)             // [1, 1, 1]
let filled = [Double].full(4, value: 3.14)  // [3.14, 3.14, 3.14, 3.14]
```

The element type is selected by the bracket notation on the left of the call. `[Double].zeros(5)` produces an array of `Double` values; `[Int].zeros(5)` produces an array of `Int` values. Quiver does not infer the type from context — the caller chooses it explicitly.

### Generating sequences

For evenly spaced sequences, `linspace` and `arange` cover the two common cases. `linspace` produces a fixed number of values between two endpoints; `arange` produces values at a fixed step size starting from a given value.

```swift
// Create evenly spaced values
let linear = [Double].linspace(start: 0, end: 10, count: 5)  // [0.0, 2.5, 5.0, 7.5, 10.0]

// Create sequences with specific step sizes
let range = [Double].arange(0, 10, step: 2.5)  // [0.0, 2.5, 5.0, 7.5]
```

The two methods differ in how they treat the upper bound. `linspace` includes both endpoints, so the count is exact and the step is whatever it needs to be. `arange` includes the start and excludes the end, so the step is exact and the count is whatever it works out to. Reach for `linspace` when the number of points matters, and `arange` when the spacing matters.

### Filling matrices with constants

The same `zeros`, `ones`, and `full` methods extend to two-dimensional arrays by taking a row count and a column count. The result is a `[[Double]]` (or `[[Int]]`, etc.) with every element set to the fill value.

```swift
// Create 2D arrays
let zeroMatrix = [Int].zeros(3, 2)  
// [[0, 0], [0, 0], [0, 0]]

let oneMatrix = [Double].ones(2, 3)  
// [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

let filledMatrix = [Int].full(2, 2, value: 7)  
// [[7, 7], [7, 7]]
```

Quiver follows the standard mathematical convention: the first dimension is rows and the second is columns. A `[Double].zeros(3, 2)` has three rows and two columns, not the other way around.

### Identity and diagonal matrices

Identity matrices appear constantly in linear algebra — they are the multiplicative identity for matrix multiplication, and they are the starting point for building transformations incrementally. Diagonal matrices generalize the same idea, with arbitrary values along the main diagonal and zeros everywhere else.

```swift
// Create an identity matrix
let identity = [Double].identity(3)  
// [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

// Create a diagonal matrix from a vector
let diag = [Int].diag([1, 2, 3])
// [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
```

A scaling transformation is exactly a diagonal matrix whose entries are the per-axis scale factors. For the role identity and diagonal matrices play in transformations, see <doc:Matrix-Transformations>.

## Common patterns

The generation methods compose with the rest of Quiver in a few recurring ways. A typical numerical pipeline starts with an empty buffer, builds an evenly spaced input, or initializes a transformation matrix that the rest of the code mutates incrementally.

### Initializing buffers

When accumulating results into an array, starting with `zeros` of the right length avoids an out-of-bounds index check on every write. Starting with `ones` is the same idea for multiplicative accumulation.

```swift
// Initialize a container for results
let results = [Double].zeros(dataPoints.count)

// Start with all ones (multiplicative identity)
let factors = [Double].ones(n)
```

### Generating plot inputs

Plotting a function across a range begins with a sequence of x-values and a `map` over the function. `linspace` is the right choice here because the number of sample points usually matters more than the exact spacing between them.

```swift
import Foundation

// Generate x-coordinates for plotting
let x = [Double].linspace(start: 0, end: 2 * Double.pi, count: 100)

// Generate y-coordinates (sine wave)
let y = x.map { sin($0) }
```

### Building transformations incrementally

Transformations often start from an identity matrix and accumulate operations onto it. This pattern is especially common in graphics work, where a single transformation may need to scale, rotate, and translate before it is applied.

```swift
// Start with an identity matrix for transformations
var transform = [Double].identity(4)

// Modify specific elements for a particular transformation
transform[0][3] = 10.0  // Add translation
```

### Memory cost

These methods allocate the entire array in a single step. For very large dimensions — millions of elements or large dense matrices — that allocation is the dominant cost of the call, and the resulting array sits in memory until it is released. The cost is the same as any equivalent manual allocation, but it is worth being aware of when generating large structures inside a tight loop.

## See also

- <doc:Matrix-Operations>
- <doc:Matrix-Transformations>
- <doc:Random-Number-Generation>

## Topics

### Basic array creation
- ``Swift/Array/zeros(_:)``
- ``Swift/Array/ones(_:)``
- ``Swift/Array/full(_:value:)``

### Sequence generation
- ``Swift/Array/linspace(start:end:count:)``
- ``Swift/Array/arange(_:_:step:)-8fjm5``

### Matrix creation
- ``Swift/Array/zeros(_:_:)``
- ``Swift/Array/ones(_:_:)``
- ``Swift/Array/full(_:_:value:)``
- ``Swift/Array/identity(_:)``
- ``Swift/Array/diag(_:)``
