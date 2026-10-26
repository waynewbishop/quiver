# Quiver — Statistics, Linear Algebra & Machine Learning in Swift

[![Swift](https://img.shields.io/badge/Swift-5.9%2B-orange)](https://swift.org)
[![Platforms](https://img.shields.io/badge/platforms-iOS%20%7C%20macOS%20%7C%20watchOS%20%7C%20tvOS%20%7C%20visionOS%20%7C%20Linux-lightgrey)](https://waynewbishop.github.io/quiver/documentation/quiver/installation/)
[![Release](https://img.shields.io/github/v/release/waynewbishop/quiver)](https://github.com/waynewbishop/quiver/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Linux Tests](https://github.com/waynewbishop/quiver/actions/workflows/linux.yml/badge.svg)](https://github.com/waynewbishop/quiver/actions/workflows/linux.yml)

Quiver is an open-source Swift package for statistics, linear algebra, and machine learning — zero external dependencies, running on every Apple platform and server-side Swift. It extends the standard `Array` type, so mathematical operations feel natural to Swift developers, and every model exposes its learned parameters: nothing is a black box. Quiver is built to teach and built to ship — the same code that explains a concept deploys to iPhone, Apple Watch, or a server.

## Who Quiver is for

**Students** working through a course, textbook, or self-study. Quiver extends the standard `Array` type, so there's no custom container to learn before the math starts — and every model exposes its learned parameters, so you can always see what the algorithm actually did. Start with the [Quiver Notebook](https://github.com/waynewbishop/quiver-notebook) or the [73-recipe Cookbook](https://github.com/waynewbishop/quiver-cookbook). Quiver runs on Linux; the Notebook is Mac-only.

**Educators** preparing a lecture, unit, or full course get material that respects the academic calendar. The Notebook launches with all datasets pre-bundled — no environment drift, no per-student setup or CSV hunting. Fork it, drop your own examples in, distribute a URL to your class.

**Apple-platform developers** who want something simple and practical get production-ready numerical code with zero dependencies and no runtime to manage. Prototype in the Notebook, then drop the same code into an iOS, watchOS, or visionOS target unchanged — models are `Codable` value types, so what you trained is what you ship. True Effort Score shows the pattern end to end: learn on the watch, save between runs, score the next one.

## Installation

Add Quiver to your project with Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/waynewbishop/quiver", from: "1.6.0")
]
```

## Quick Start

### Score a Run (True Effort Score)

```swift
import Quiver

var tes = TrueEffortScore()   // first run: no personal history yet

// Record each sample, live or read back from HealthKit:
tes.record(heartRate: 152, pace: 5.4, cadence: 174, grade: 1.2,
           verticalOscillation: 7.9, altitude: 100, at: sampleDate)

if let result = tes.finalize() {   // nil if the run is too short to keep
    result.adjusted                // the headline score
    result.meanResidual            // observed − expected HR, in bpm
    result.effortDistribution      // share of time in each category
}
```

### Train a Model

```swift
import Quiver

// Training data: square footage → price
let features: [[Double]] = [[1000], [1500], [2000], [2500], [3000]]
let targets = [150000.0, 200000.0, 260000.0, 310000.0, 370000.0]

// Fit, predict, evaluate — three lines
let model = try LinearRegression.fit(features: features, targets: targets)
let predictions = model.predict(features)
let r2 = predictions.rSquared(actual: targets)  // 0.99+

// Generate a trend line for charting
let trendX = Array.linspace(start: 500.0, end: 3500.0, count: 50)
let trendY = model.predict(trendX)  // parallel arrays ready for Swift Charts
```

### Vector Operations

```swift
import Quiver

let v = [3.0, 4.0]
v.magnitude          // 5.0
v.normalized         // [0.6, 0.8]

let force = [5.0, 3.0, 2.0]
let displacement = [10.0, 0.0, 0.0]
let work = force.dot(displacement)  // 50.0

let product1 = [4.2, 7.8, 3.1, 9.5]
let product2 = [3.8, 8.2, 2.9, 9.7]
product1.cosineOfAngle(with: product2)  // 0.999 (very similar)
```

### Statistics and Data Analysis

```swift
import Quiver

let sales = [45.0, 52.0, 48.0, 61.0, 55.0, 58.0, 49.0, 67.0, 72.0, 69.0]

sales.mean()                 // 57.6
sales.median()               // 56.5
sales.standardDeviation()    // 9.4 (sample, ddof: 1 by default)

let smoothed = sales.rollingMean(window: 3)
let outliers = sales.outlierMask(threshold: 1.2).trueIndices
```

## Featured: True Effort Score (new in 1.6.0)

**True Effort Score (TES)** is a transparent training-load model for Apple Watch, built entirely with Quiver. Traditional scores reduce a run to one signal — heart rate or estimated power. TES reads six signals per moment (heart rate, pace, cadence, grade, vertical oscillation, altitude), classifies each moment from **Recovery** to **Hard**, and prices the efforts single-signal systems miss: steep descents, intervals, technical terrain, power-hikes. A personal baseline learns the heart rate your workload should produce, so heat and cardiac drift surface as a residual instead of inflating your score.

→ [How TES works](https://waynewbishop.github.io/quiver/documentation/quiver/true-effort-score/) · [Research whitepaper (PDF)](https://waynewbishop.github.io/quiver/papers/true-effort-score.pdf) · [API reference](https://waynewbishop.github.io/quiver/documentation/quiver/trueeffortscore/)

## Capabilities

- **Vector & matrix math** — element-wise arithmetic, dot product, normalization, projections; multiplication, transpose, inverse, determinant, LU decomposition, eigendecomposition, covariance matrix, log-determinant, numerical Jacobian
- **Similarity & distance** — cosine similarity (single and batch), six distance metrics, Top-K selection, duplicate detection, cluster analysis
- **Statistics** — central tendency and dispersion, typed summaries, correlation matrices, shape diagnostics, outlier detection, normal distribution functions, bootstrap resampling with confidence intervals, Bayes' rule, contingency tables, empirical rule
- **Data preparation** — `Panel` named-column data, `Pipeline` (scaler + model in one value type), feature scaling, train/test split, k-fold cross-validation, stratified splitting, imbalance handling
- **Machine learning** — Naive Bayes, K-Nearest Neighbors, Linear/Logistic/Ridge Regression, Gradient Descent, K-Means, PCA (also a `Pipeline` reducer stage), `ResidualModel` — consistent `fit`/`predict` API, fully inspectable parameters, all models `Codable`
- **Signal processing** — Fourier transforms and power spectrum analysis
- **Text & embeddings** — tokenization, the `Embedder` protocol, `EmbeddingIndex` retrieval, `Chunker`
- **Evaluation** — confusion matrix, per-class classification report, R², MSE, RMSE
- **Visualization-ready data** — trend lines, elbow curves, confusion-matrix heatmaps, softmax probability bars, downsampling, stacked series
- **Algebra utilities** — polynomials and `polyfit`, linear-system solver, numerical integration, exact `Fraction` arithmetic, math rendering

Full details live in the [documentation](https://waynewbishop.github.io/quiver/documentation/quiver/).

## Learn Quiver

Quiver is a teaching platform as much as a framework. Start wherever you are:

- **[Quiver Notebook](https://github.com/waynewbishop/quiver-notebook)** — browser-based Swift editor with teaching datasets pre-loaded.
- **[Cookbook](https://github.com/waynewbishop/quiver-cookbook)** — 73 interactive recipes, each a single `.swift` file for the Xcode `#Playground` macro.
- **[Documentation & primers](https://waynewbishop.github.io/quiver/documentation/quiver/)** — the concepts behind the code, plus full API reference.
- **[Swift Algorithms & Data Structures](https://waynewbishop.github.io/swift-algorithms/)** — the free companion book for deeper foundations.

## Design Philosophy

* **Swift-first**: Extends standard Swift arrays — no custom container types, no conversion overhead. Students learn foundational programming and numerical computing together, on native `[Double]` from day one
* **Transparent ML**: Every model exposes its learned parameters — inspect coefficients, centroids, and class statistics. True Effort Score takes this furthest: the entire training-load model is open, auditable, and reproducible
* **Educational**: Clear implementations that map directly to mathematical concepts
* **Zero dependencies**: Pure Swift, no external frameworks required — no environment drift for students, no deployment surprises for apps

## Documentation

Full API documentation at [waynewbishop.github.io/quiver](https://waynewbishop.github.io/quiver/documentation/quiver/), including:
* True Effort Score — model guide and [research whitepaper](https://waynewbishop.github.io/quiver/papers/true-effort-score.pdf)
* Vector operations and linear algebra primer
* Statistical function reference
* Matrix operations and transformations
* Data preparation and visualization guides
* Machine learning model guides (Naive Bayes, K-Nearest Neighbors, Linear, Logistic & Ridge Regression, Gradient Descent, K-Means, PCA, Residual Model)

Quiver is the companion numerical computing package for [Swift Algorithms & Data Structures](https://waynewbishop.github.io/swift-algorithms/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Quiver is available under the Apache License, Version 2.0. See the LICENSE file for more info.

## Newsletter

[The Feature Vector](https://featurevector.substack.com) — a newsletter about ML intuition for engineers, built in Swift. One idea per issue, with a recipe you can run in Xcode.

## Questions

Have a question? Feel free to contact me on [LinkedIn](https://www.linkedin.com/in/waynebishop).
