# Exploring Quiver with Xcode

Using the playground macro to inspect Quiver values.

## Overview

The `#Playground` macro, introduced in [Xcode 26](https://developer.apple.com/xcode/), turns any Swift file inside a project into an interactive surface — write an expression, see the result inline in the Canvas, no build-and-run cycle. For a project that already depends on Quiver, this is the most direct way to inspect a value, verify a calculation, or sanity-check a method without leaving the codebase.

> Tip: The `#Playground` macro is not the same as a `.playground` file. Traditional `.playground` files run in an isolated sandbox and cannot import Swift packages. The `#Playground` macro compiles as part of the project, so it has full access to SPM dependencies, including Quiver, with no extra configuration.

### Writing a playground inside a project

Add a new Swift file to the project, import `Playgrounds`, and wrap the code in a `#Playground` block. The block compiles and runs as part of the project, so it can use any Quiver type the project already has access to:

```swift
import Playgrounds
import Quiver

#Playground {
    // Create a 2D vector and compute its length: sqrt(3² + 4²)
    let v = [3.0, 4.0]
    let length = v.magnitude  // 5.0

    // Divide each element by the length to get a unit vector
    let unit = v.normalized  // [0.6, 0.8]

    // Display the unit vector as exact fractions
    let fractions = unit.asFractions()  // [3/5, 4/5]
    print("length: \(length), unit: \(fractions)")
}
```

The `magnitude` property measures how long the vector is, specifically the distance from the origin to the point `[3, 4]`. The `normalized` property divides each element by that magnitude, producing a unit vector that preserves direction but has a length of exactly 1. The `asFractions` method shows the result as rational numbers, which is useful for verifying exact values.

### Multiple experiments in one file

Name the playground blocks to organize separate inspections. Each block runs independently:

```swift
import Playgrounds
import Quiver

#Playground("Dot Product") {
    // Sum of element-wise products: (1×4) + (2×5) + (3×6)
    let a = [1.0, 2.0, 3.0]
    let b = [4.0, 5.0, 6.0]
    let dotProduct = a.dot(b)  // 32.0

    // Normalize by both magnitudes to get cosine similarity
    let cosine = dotProduct / (a.magnitude * b.magnitude)
    print(cosine)  // 0.97
}

#Playground("Cosine Similarity") {
    // Compare a search query against a document vector
    let query = [0.8, 0.7, 0.2]
    let doc = [0.7, 0.6, 0.3]

    // Measures directional alignment, ignoring magnitude (1.0 = identical)
    let similarity = query.cosineOfAngle(with: doc)  // 0.99
    print(similarity)
}
```

Both blocks compute cosine similarity — one manually, one with a convenience method. The first block shows the underlying math: compute the dot product, divide by both magnitudes. The second block does the same calculation in a single call with `cosineOfAngle(with:)`. Named blocks are useful for comparing related operations side by side or working through a series of inspections in a single file.

### Inspecting results

The Canvas shows each variable's value as we write it. For arrays and matrices, this makes it easy to verify intermediate steps without inserting `print` statements:

```swift
import Playgrounds
import Quiver

#Playground("Matrix Operations") {
    let A = [[1.0, 2.0], [3.0, 4.0]]
    let B = [[5.0, 6.0], [7.0, 8.0]]

    // Each element in the result is a dot product of a row from A and a column from B
    let product = A.multiplyMatrix(B)  // [[19.0, 22.0], [43.0, 50.0]]

    // Transpose the result so row 0 becomes column 0
    let flipped = product.transpose()  // [[19.0, 43.0], [22.0, 50.0]]

    // Check whether the transposed result is invertible (non-zero determinant)
    let det = flipped.determinant  // 4.0
    print(det)
}
```

Matrix multiplication combines two matrices by computing dot products between rows and columns. The transpose flips a matrix along its diagonal, turning rows into columns. The determinant is a single number that captures whether a matrix is invertible.

### Iterating on parameters

The `#Playground` macro re-evaluates automatically on edit, which makes it useful for tuning a value against the project's real types. Adjust a feature range, swap a scaling strategy, or change a model parameter, and the result updates without restarting:

```swift
import Playgrounds
import Quiver

#Playground("Comparing Normalization") {
    let raw = [100.0, 200.0, 300.0, 400.0, 500.0]

    // Rescale to [0, 1] based on min and max values
    let minMax = raw.scaled(to: 0.0...1.0)  // [0.0, 0.25, 0.5, 0.75, 1.0]

    // Center around zero with unit standard deviation
    let zScore = raw.standardized()  // [-1.41, -0.71, 0.0, 0.71, 1.41]

    // Compare the two approaches side by side
    if let minMaxMean = minMax.mean(), let zScoreMean = zScore.mean() {
        let comparison = minMaxMean - zScoreMean
        print(comparison)  // 0.5
    }
}
```

Min-max scaling compresses values into a target range. Z-score scaling rescales data so every value is measured in standard deviations from the mean — see <doc:Statistics-Primer> for the concept. Both are useful for inspecting how a transformation behaves before committing to one in production code. For ML pipelines where training and test data must be scaled consistently, use `FeatureScaler` with the fit-then-transform pattern — see <doc:Feature-Scaling>.

### Related
- <doc:Quiver-Notebook>
- <doc:Notebook-Datasets>
- <doc:Statistics-Primer>
- <doc:Feature-Scaling>
