# Usage

Explore Quiver interactively using the Xcode Playground macro.

## Overview

Quiver is a Swift package for numerical computing that covers vectors, matrices, statistics, and machine learning. It runs on all Apple platforms including Linux and server-side frameworks like Vapor. But production code is only half the story. Interactive environments let us write a line of code, see the result immediately, and build intuition one expression at a time. The `#Playground` macro, introduced in [Xcode 26](https://developer.apple.com/xcode/), brings that interactive workflow to any Swift project, including projects that depend on Quiver.

A playground is a space for exploration. We write an expression, assign it to a variable, and the Canvas shows the result inline. There is no build-and-run cycle and no separate output window. For a library like Quiver, this means we can compute a dot product, inspect a matrix inverse, or fit a regression model and see every intermediate value as we type. It is the fastest way to learn what an operation does, verify that the math is correct, or prototype an idea before committing to an implementation.

> Tip: The `#Playground` macro is not the same as a `.playground` file. Traditional `.playground` files run in an isolated sandbox and cannot import Swift packages. The `#Playground` macro compiles as part of the project, so it has full access to SPM dependencies, including Quiver, with no extra configuration.

### Writing your first playground

Add a new Swift file to any project that already depends on Quiver, import `Playgrounds`, and wrap the code in a `#Playground` block:

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

Name the playground blocks to organize experiments. Each block runs independently:

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

Both blocks compute cosine similarity — one manually, one with a convenience method. The first block shows the underlying math: compute the dot product, divide by both magnitudes. The second block does the same calculation in a single call with `cosineOfAngle(with:)`. The dot product multiplies corresponding elements and sums the results, and normalizing by both magnitudes produces a value between -1 and 1 that measures how closely two vectors point in the same direction, regardless of their length.

Named blocks are useful for comparing related operations side by side or working through a series of exercises in a single file.

### Inspecting results

The Canvas shows each variable's value as we write it. For arrays and matrices, this makes it easy to verify intermediate steps:

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

Matrix multiplication combines two matrices by computing dot products between rows and columns. The transpose flips a matrix along its diagonal, turning rows into columns. The determinant is a single number that captures whether a matrix is invertible. A zero determinant means the matrix collapses space into a lower dimension and cannot be inverted. These three operations are the building blocks for solving systems of equations, fitting regression models, and applying geometric transformations.

### Iterating on code

The `#Playground` macro re-evaluates automatically when we edit. This makes it ideal for experimenting with different parameters and seeing the effect immediately. We can adjust feature ranges, change scaling strategies, or compare normalization approaches without restarting.

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

Min-max scaling compresses values into a target range. Z-score scaling rescales data so every value is measured in standard deviations from the mean — see <doc:Statistics-Primer> for the concept. Both are useful for normalizing data for charts and exploration. For ML pipelines where training and test data must be scaled consistently, use `FeatureScaler` with the fit-then-transform pattern — see <doc:Feature-Scaling>.

### Signal processing

Quiver also operates in the frequency domain. The Fourier transform reveals which repeating cycles are hidden inside a signal — a fundamentally different question from regression or classification, but one that uses the same `[Double]` arrays and the same interactive workflow:

```swift
import Playgrounds
import Quiver

#Playground("Detect a Frequency") {
    // Generate a 440 Hz tone at 8000 Hz sample rate
    let sampleRate = 8000.0
    let signal = [Double].sineWave(frequency: 440.0, sampleRate: sampleRate, count: 256)

    // Find the dominant frequency — one call, automatic padding
    let dominant = signal.fourierDominantFrequency(sampleRate: sampleRate, windowed: true)
    dominant  // 437.5 Hz (nearest bin center)
}
```

For the full Fourier API — spectrum computation, windowing, phase analysis, and real-world applications — see <doc:Fourier-Transform>.

> Tip: The [Quiver Cookbook](https://github.com/waynewbishop/quiver-cookbook) has interactive recipes that run as `#Playground` macros in Xcode. Each recipe solves a single problem — from computing distance to training a classifier — in under 30 lines.

> Tip: For an Xcode-free interactive workflow, the <doc:Quiver-Notebook> runs the same Swift code in a browser tab. Useful for classroom labs, machines without Xcode installed, and quickly trying a snippet without creating a project.
