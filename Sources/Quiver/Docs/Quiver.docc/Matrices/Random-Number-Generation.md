# Random Number Generation

Generate arrays of random values for testing, simulation, and initialization.

## Overview

Quiver provides methods to generate arrays filled with random values. These functions support uniform distributions (default or custom range), normal (Gaussian) distributions, and random integers — the three most common random generation patterns in numerical computing.

### Uniform random arrays

Create arrays with uniformly distributed random values:

```swift
import Quiver

// Generate a 1D array of 5 random values between 0 and 1
let randomValues = [Double].random(5)
// Example output: [0.12, 0.87, 0.43, 0.59, 0.22]

// Generate a 2D array (3×2 matrix) of random values
let randomMatrix = [Double].random(3, 2)
// Example output:
// [[0.31, 0.95],
//  [0.47, 0.72],
//  [0.13, 0.84]]
```

> Note: Each call produces different random values. The examples above show possible outputs.

### Custom range

Generate values in any range using the `in:` parameter:

```swift
// Random values between -1 and 1
let centered = [Double].random(5, in: -1.0...1.0)
// Example: [-0.34, 0.71, -0.89, 0.12, 0.55]

// Random temperatures between -10 and 40
let temperatures = [Double].random(7, in: -10.0...40.0)
// Example: [12.5, -3.2, 28.7, 5.1, 35.8, -8.4, 19.3]

// 2D matrix with custom range
let data = [Double].random(3, 4, in: 0.0...100.0)
```

### Normal distribution

Generate values from a Gaussian (bell curve) distribution using the Box-Muller transform:

```swift
// Standard normal: mean = 0, std = 1
let standardNormal = [Double].randomNormal(1000)

// Custom mean and standard deviation
let heights = [Double].randomNormal(500, mean: 170.0, std: 10.0)
// Example: values clustered around 170, mostly between 150-190

// 2D matrix of normal values
let noiseMatrix = [Double].randomNormal(3, 4, mean: 0.0, std: 0.1)
```

### Random integers

Generate arrays of random integers in a half-open range:

```swift
// Random integers from 0 to 9
let digits = [Int].random(10, in: 0..<10)
// Example: [3, 7, 1, 9, 0, 4, 6, 2, 8, 5]

// Random dice rolls (1 to 6)
let diceRolls = [Int].random(100, in: 1..<7)

// 2D matrix of random integers
let labels = [Int].random(5, 3, in: 0..<4)
```

### Float and double support

All uniform and normal generation methods work with both `Float` and `Double` types:

```swift
// Float variants
let floatValues = [Float].random(3)
let floatNormal = [Float].randomNormal(100, mean: 0.0, std: 1.0)
let floatRange = [Float].random(5, in: 0.0...10.0)
```

## Common use cases

### Random vectors and transformations

Generate random vectors and apply the matrix transformations covered in <doc:Matrix-Transformations>:

```swift
// Create a random 2D position
let position = [Double].random(2, in: -10.0...10.0)
// Example: [3.7, -6.2]

// Apply a 90° rotation
let rotate90 = [
    [0.0, -1.0],
    [1.0,  0.0]
]
let rotated = position.transformedBy(rotate90)
```

### Testing statistical operations

Generate datasets with known characteristics to verify Quiver's statistics functions:

```swift
// Uniform data — mean should be near the midpoint
let uniform = [Double].random(100, in: 50.0...70.0)
if let avg = uniform.mean(), let std = uniform.std() {
    print(avg)  // Approximately 60.0
    print(std)  // Approximately 5.8
}

// Normal data — verify the distribution matches expectations
let normal = [Double].randomNormal(1000, mean: 100.0, std: 15.0)
if let median = normal.median() {
    print(median)  // Approximately 100.0
}
```

### Similarity with random embeddings

Generate random embedding vectors to test the similarity operations from <doc:Similarity-Operations>. For a complete text-to-results pipeline using real embeddings, see <doc:Semantic-Search>.

```swift
// Simulate three document embeddings (3 dimensions each)
let docs = [Double].random(3, 3, in: -1.0...1.0)
let query = [Double].random(3, in: -1.0...1.0)

// Rank documents by cosine similarity to the query
let scores = docs.cosineSimilarities(to: query)
let ranked = scores.topIndices(k: 2)
```

### Random matrix operations

Generate matrices to explore determinants and invertibility from <doc:Determinants-Primer>:

```swift
// Random 2×2 matrix — most will be invertible
let matrix = [Double].random(2, 2, in: -5.0...5.0)
let det = matrix.determinant

if det != 0.0 {
    let inverse = try matrix.inverted()
    // Verify: matrix × inverse ≈ identity
    let identity = matrix.multiplyMatrix(inverse)
}
```

### Implementation details

Uniform random generation uses Swift's built-in `Double.random(in:)` and `Float.random(in:)` functions. Normal distribution values are generated using the Box-Muller transform, which converts pairs of uniform random values into independent standard normal samples. Integer generation uses Swift's `Int.random(in:)` with a half-open range.

## Topics

### Uniform random generation
- ``Swift/Array/random(_:)-6ulik``
- ``Swift/Array/random(_:_:)-9gsef``

### Normal distribution
- ``Swift/Array/randomNormal(_:mean:std:)->[Double]``
- ``Swift/Array/randomNormal(_:_:mean:std:)->[[Double]]``
