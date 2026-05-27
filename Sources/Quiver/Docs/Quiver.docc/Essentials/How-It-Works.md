# How It Works

Understanding Quiver's architecture as a layer over the Swift array type.

## Overview

Quiver adds numerical computing methods directly to Swift's standard `Array` type using a language feature called **extensions**. Rather than introducing custom container types, Quiver works with standard Swift `Array`. The result is that a plain `[Double]` gains mathematical operations without becoming a different type.

```swift
import Quiver

// This is a standard Swift Array<Double> — nothing special
let position = [3.0, 4.0]

// Quiver adds .magnitude directly to the array
position.magnitude  // 5.0
```

[Magnitude](<doc:Linear-Algebra-Primer>) measures how far a point sits from the origin, the length of the vector. In this example, the `position` array could be passed to SwiftUI, Swift Charts, SwiftData, or any other Swift API.

### What are extensions

An **extension** in Swift adds new methods, computed properties, and initializers to a type that already exists, even one we did not write. The capability is added directly to the type, and every instance gains it automatically. Swift's standard library uses this same mechanism extensively — methods like `.sorted()`, `.reversed()`, and `.contains()` are all added to `Array` through extensions. Quiver follows the same pattern to add numerical operations.

> Note: Extensions can add new functionality to a type, but they cannot override or modify existing behavior. Quiver's extensions are purely additive; they never change how `Array` already works. To learn more about Swift's type system, generics, and protocol-oriented architecture see [Swift Algorithms & Data Structures](https://waynewbishop.github.io/swift-algorithms/), the companion book to this framework.

### Constrained extensions

Not every array operation makes sense for every element type. Computing a mean requires division, which integers cannot do precisely. Computing magnitude requires a square root, which strings cannot produce. Quiver needs a way to say "this method only exists when the elements are the right kind of number."

Swift solves this with **constrained extensions**, an extension that applies only when the generic type parameter meets a specific requirement. The constraint acts as a gate: if the elements qualify, the methods appear. If they do not, the compiler prevents us from calling them:

```swift
let integers = [1, 2, 3, 4, 5]
let doubles = [3.0, 4.0, 5.0]

// These compile — the constraints are satisfied
integers.sum()           // Numeric ✓
doubles.mean()           // FloatingPoint ✓

// These do NOT compile — the compiler catches the error
integers.mean()          // Int is not FloatingPoint
integers.magnitude       // Int is not FloatingPoint
```

Each failed call is caught before the code ever runs. The compiler tells us exactly which protocol the element type is missing. When we refactor an array from `[Double]` to `[Int]`, the build itself flags every Quiver call that no longer applies.

Swift's type system also encodes dimensionality. The compiler distinguishes between `[Double]` (a vector) and `[[Double]]` (a matrix) at compile time, so there is no need for a runtime property to query how many dimensions an array has — the type signature already tells us. For a detailed look at `.shape`, `.size`, and working with matrix dimensions, see <doc:Shape-And-Size>.

### Models are always ready

Quiver's ML models are created fully trained. There is no separate "empty" or "unfitted" state. The `fit` method returns a model that is immediately ready to use:

```swift
import Quiver

let model = KNearestNeighbors.fit(features: trainingData, labels: labels, k: 3)
let predictions = model.predict(newData)
```

Because `fit` is the only way to create a model, the compiler makes it impossible to call `predict` on something that has not been trained. There is no runtime error for "model not fitted"; the situation cannot arise. Every model in Quiver follows this pattern including `LinearRegression`, `GaussianNaiveBayes`, `KNearestNeighbors`, and `KMeans`.

Models are also immutable. Once created, their coefficients, centroids, and learned parameters cannot change. This eliminates an entire category of bugs where a model is accidentally retrained or modified between predictions.

> Note: Models also conform to `Codable`. Because every stored property is a basic Swift type (arrays of numbers, integers, and booleans), the compiler auto-synthesizes JSON encoding and decoding. A model trained once can be saved to disk and loaded on the next app launch without retraining. See <doc:Model-Persistence> for platform-specific guidance.

### Clean output by default

Every model and result type produces a readable summary when printed. This makes Playground exploration and debugging straightforward; there is no wall of raw properties to parse:

```swift
print(model)    // KNearestNeighbors: k=3, euclidean, 6 training points, 2 features
print(cluster)  // Cluster: center [1.23, 1.97], 3 points
print(cm)       // TP: 3  FP: 1  TN: 3  FN: 1  (accuracy: 75.0%)
```

### Typed summary returns

When a Quiver method needs to return several related values at once, it returns a typed value rather than a dictionary or an anonymous tuple. The `Quartiles`, `ColumnSummary`, `PanelSummary`, and `RegressionSummary` types are the patterns we will see repeatedly.

```swift
let summary: PanelSummary = panel.summary()

print(summary) // a formatted table

if let price = summary.columns["price"] {
    print(price.mean) // named field, compile-time checked
}
```

### A focused and intentional scope

Quiver is designed for educational use, on-device computing, and data science workflows where understanding the mathematics matters as much as the result. GPU acceleration, automatic differentiation, and distributed training are outside that scope. Each brings external dependencies, platform restrictions, and a steeper learning curve that works against the framework's goals of clarity, portability, and zero-dependency deployment.

### Performance characteristics

The design prioritizes clarity and portability: the same code runs identically on macOS, iOS, watchOS, visionOS, and Linux. Most operations (vector arithmetic, statistics, broadcasting, boolean masking, element-wise math) are linear and scale predictably to millions of elements. Fourier analysis is `O(n log n)` and scales efficiently to tens of thousands of samples.

> Important: The operations worth thinking about are matrix multiplication, matrix inversion, the determinant, and pairwise comparisons like `findDuplicates(threshold:)`, `clusterCohesion`, and `correlationMatrix()`. These grow quadratically or cubically with input size, so they perform well for the hundreds-to-low-thousands range typical in educational and on-device use cases.

For larger inputs, many of these operations can be implemented to run across multiple CPU cores. The <doc:Concurrency-Primer> shows how.
