# How It Works

Understanding Quiver's architecture as a layer over the Swift array type.

## Overview

Quiver adds numerical computing methods directly to Swift's standard `Array` type using a language feature called **extensions**. Rather than introducing custom container types, Quiver works with standard Swift `Array`, with a small number of value types reserved for objects that carry their own algebra — ``Polynomial`` is one such example. The result is that a plain `[Double]` gains mathematical operations without becoming a different type.

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

Because `fit` is the only way to create a model, the compiler makes it impossible to call `predict` on something that has not been trained. There is no runtime error for "model not fitted"; the situation cannot arise. Every model in Quiver follows this pattern including ``LinearRegression``, ``GaussianNaiveBayes``, ``KNearestNeighbors``, and ``KMeans``.

Models are also immutable. Once created, their coefficients, centroids, and learned parameters cannot change. This eliminates an entire category of bugs where a model is accidentally retrained or modified between predictions.

> Note: Models also conform to `Codable`. Because every stored property is a basic Swift type (arrays of numbers, integers, and booleans), the compiler auto-synthesizes JSON encoding and decoding. A model trained once can be saved to disk and loaded on the next app launch without retraining. See <doc:Model-Persistence> for platform-specific guidance.

### Shared behavior through protocols

Quiver's models share their prediction interface through two protocols. The ``Regressor`` protocol guarantees that a model predicts continuous values, and the ``Classifier`` protocol guarantees that a model predicts class labels, so any code written against a protocol works for every model that adopts it. The guarantee lives in the type system, which means a model that fails to provide `predict` does not compile.

Conforming models gain the protocol's prediction surface, including a scalar convenience for single-feature models that the protocol supplies once for the whole family:

```swift
import Quiver

// LinearRegression is a Regressor, so it predicts continuous values
let model = try LinearRegression.fit(features: sqft, targets: prices)
model.predict([[3500.0]])  // an array of predictions

// The Regressor protocol supplies a scalar overload for single-feature models
model.predict(3500.0)      // one value in, one value out
```

> Note: Writing one function against ``Regressor`` lets it accept ``LinearRegression``, ``Ridge``, and ``GradientDescent`` interchangeably, and writing against ``Classifier`` accepts ``GaussianNaiveBayes``, ``KNearestNeighbors``, and ``LogisticRegression`` the same way. The protocol is the contract every model in its family keeps.

### Failure is in the return type

Every model's `fit` method states in its signature whether training can fail. A method that throws can fail and forces us to handle the failure, a method that returns a plain value cannot fail at all, and a method that returns an optional reports failure by handing back `nil`. The compiler reads that signature, so the obligation to handle a bad fit is checked before the program ever runs.

The three shapes sit side by side, and each one tells us at the call site exactly how much care the result demands:

```swift
import Quiver

// Throwing fit — training can fail, so the call must handle it
let regression = try LinearRegression.fit(features: sqft, targets: prices)

// Plain-value fit — training cannot fail, so there is nothing to handle
let classifier = KNearestNeighbors.fit(features: points, labels: labels, k: 3)

// Optional fit — failure is reported as nil
if let curve = [Double].polyfit(x: xs, y: ys, degree: 2) {
    curve.coefficients  // the fit succeeded
}
```

> Tip: When a `fit` call returns a plain value rather than throwing or returning an optional, the type is telling us the operation is total — there is no failure path to write, because none exists. ``GaussianNaiveBayes``, ``KNearestNeighbors``, and ``StandardScaler`` follow this pattern.

### Clean output by default

Every model and result type produces a readable summary when printed. This makes Playground exploration and debugging straightforward; there is no wall of raw properties to parse:

```swift
print(model)    // KNearestNeighbors: k=3, euclidean, 6 training points, 2 features
print(cluster)  // Cluster: center [1.23, 1.97], 3 points
print(cm)       // TP: 3  FP: 1  TN: 3  FN: 1  (accuracy: 75.0%)
print(p)        // 2x² + 3x + 1
print(f)        // 3/5
```

### Presentation-only types

Quiver computes in `Double` and renders in the form a reader can read. Two methods anchor the pattern: `asFraction` returns rational structure as a real ``Fraction`` value, and `asExpression` returns a Unicode-formatted string ready to display. They compose — chain `asFractions().asExpression()` to see the rational form of a vector or matrix as a bracketed block — and the underlying numeric values are never touched.

```swift
let v = [0.6, 0.75, 0.5]
print(v.asFractions().asExpression())
// ⎡ 3/5 ⎤
// ⎢ 3/4 ⎥
// ⎣ 1/2 ⎦
```

See <doc:Rendering-Math-Primer> for the full catalog: the scalar, vector, and matrix forms; the column-default vector convention; the `relativeZeroTolerance` parameter for fitted polynomials; the edge-case rules for `NaN`, `±∞`, and sub-millisecond magnitudes.

### Typed summary returns

When a Quiver method needs to return several related values at once, it returns a typed value rather than a dictionary or an anonymous tuple. The ``Quartiles``, ``ColumnSummary``, ``PanelSummary``, ``RegressionSummary``, ``ClassificationReport``, ``ConfidenceInterval``, ``EmpiricalRule``, ``ContingencyTable``, and the ``BayesPrior`` / ``BayesLikelihood`` / ``BayesPosterior`` trio are the patterns we will see repeatedly.

```swift
let summary: PanelSummary = panel.summary()

print(summary) // a formatted table

if let price = summary.columns["price"] {
    print(price.mean) // named field, compile-time checked
}
```

### Designed to feed other frameworks

Quiver is built to plug into the frameworks at the center of a developer's toolkit. It shapes its output to match what is needed, so the numerical result drops straight into Swift Charts, a search ranker, or an on-device model and that framework takes it from there.

Three parts of the framework show this in action. In <doc:Data-Visualization>, Quiver computes every field a chart needs — heatmap tuples, the five numbers of a box plot, stacked series — as plain values that map straight onto Swift Charts marks, ready to draw. In <doc:Embedding-Sources>, the ``Embedder`` protocol names a single operation, text to vector, so a hand-built table, an on-device sentence model, or a custom model can all feed the same ranking surface through one method.

The retrieval pipeline in <doc:Retrieving-Context-For-Generation> is the principle taken end to end. Quiver supplies the scaffolding that turns a document and a question into a ready-to-use answer source: chunk the text, index the fragments, rank them by meaning, and assemble the context block a language model reads from. An embedding source feeds the vectors in and a language model takes the context block out, and Quiver builds the structure that connects them — the indexing and ranking that make the whole pipeline work. Every seam is a plain value, a `String` or a `[Double]`, so the same retrieval code carries cleanly across an on-device model, a different one, or a server. Quiver is the connective scaffolding, and it travels wherever the code does.

> Note: Shaping output to a known contract is what makes a source swappable. Because the ranking and rendering depend on the contract rather than on any one source, moving from a teaching baseline to a production model is a one-line change, and the work that prepared the data carries straight over.

### A focused and intentional scope

Quiver is designed for educational use, on-device computing, and data science workflows where understanding the mathematics matters as much as the result. Quiver provides analytic derivatives for polynomials, sample-based derivatives for sequences, and iterative optimization through ``GradientDescent``; what it does not provide is reverse-mode automatic differentiation over arbitrary computation graphs. GPU acceleration and distributed training are similarly outside that scope. Each brings external dependencies, platform restrictions, and a steeper learning curve that works against the framework's goals of clarity, portability, and zero-dependency deployment.

> Note: This focus shows up in the package manifest as well — Quiver depends on nothing beyond the Swift standard library and Foundation. There is no third-party package to resolve, no version graph to keep in sync, and no supply chain to audit before a build can proceed, so a Quiver project built today resolves identically a year from now. See <doc:Numerical-Literacy> for the related guarantee that numeric results are reproducible across platforms.

### Performance characteristics

The design prioritizes clarity and portability: the same code runs identically on macOS, iOS, watchOS, visionOS, and Linux. Most operations (vector arithmetic, statistics, broadcasting, boolean masking, element-wise math) are linear and scale predictably to millions of elements. Fourier analysis is `O(n log n)` and scales efficiently to tens of thousands of samples.

> Important: The operations worth thinking about are matrix multiplication, matrix inversion, the determinant, and pairwise comparisons like `findDuplicates(threshold:)`, `clusterCohesion`, and `correlationMatrix()`. These grow quadratically or cubically with input size, so they perform well for the hundreds-to-low-thousands range typical in educational and on-device use cases.

For larger inputs, many of these operations can be implemented to run across multiple CPU cores. The <doc:Concurrency-Primer> shows how.
