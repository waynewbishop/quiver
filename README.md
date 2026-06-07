# Quiver — A Swift package for statistics, linear algebra, and machine learning

Quiver expands the Swift ecosystem with a pure, Swift-first approach to statistics, linear algebra, and machine learning. By building directly on Swift's powerful type system and syntax, Quiver creates an intuitive bridge between traditional array operations and advanced mathematical concepts. Built as an extension on the standard `Array` type, the framework embraces Swift's emphasis on readability and expressiveness, offering mathematical operations that feel natural to iOS and macOS developers.

As a pure Swift library with zero external dependencies, Quiver runs on every Apple platform — iOS, macOS, watchOS, tvOS, and visionOS — as well as server-side Swift with frameworks like Vapor, Linux environments, and containerized deployments.

## Features

* **Vector Operations**
  * Element-wise arithmetic (+, -, *, /)
  * Dot product, magnitude, normalization, distance
  * Angle calculations and vector projections
  * Matrix operations (multiplication, transpose, inverse, determinant)

* **Similarity and Distance**
  * Cosine similarity (single and batch)
  * Euclidean distance
  * Top-K selection with labels
  * Duplicate detection and cluster analysis

* **Statistics**
  * Central tendency (mean, median, min, max, argmin, argmax)
  * Dispersion (variance, standard deviation, standard error, quartiles, percentiles) — `ddof=1` sample-statistics convention by default
  * Typed snapshots — `[Double].summary()` returns a `ColumnSummary` with the nine-field descriptive picture; `Panel.summary()` returns a `PanelSummary` keyed by column name
  * Pearson correlation — pairwise `correlation(with:)` on two arrays, matrix-wide `correlationMatrix()` on `[[Double]]`, labeled `Panel.correlationMatrix()`
  * Cumulative operations (sum, product)
  * Outlier detection (z-score method)
  * Shape diagnostics — `skewness()` and `kurtosis()`, paired with `skewnessReport()` which flags when an outlier-sensitive measure disagrees with an outlier-resistant one
  * Probability distributions — normal `pdf`, `logPDF`, `cdf`, and `quantile` via `Distributions.normal`
  * Random sampling — `sample(_:replace:seed:)` draws a single reproducible subset of any array, with or without replacement
  * Resampling for inference — `resampled` for the bootstrap distribution of any statistic, paired with `percentileCI` for confidence intervals

* **Array Generation**
  * Generate arrays (zeros, ones, linspace, arange, random uniform/normal)
  * Special matrices (identity, diagonal)
  * Reshape and flatten (1D ↔ 2D)
  * Broadcasting (scalar, vector, custom)

* **Algebra**
  * `Polynomial` value type with evaluation, derivative, arithmetic (`+`, `*`), and trimming
  * `polyfit(x:y:degree:)` for least-squares polynomial regression
  * `solve(_:)` for linear systems `Ax = b`
  * Numerical integration — `trapezoidalIntegral(dt:)` for the area under a sampled curve (total from a rate), with a running `cumulativeTrapezoidal(dt:)`
  * Math rendering — `asExpression()` turns arrays, matrices, and polynomials into legible math (`2x² + 3x + 1`) for teaching and console inspection
  * `Fraction` type for exact rational representation of matrix and vector results

* **Data Preparation**
  * Panel — named-column data structure with `summary()`, `unique()`, `valueCounts()`, `sortedBy()`, `correlationMatrix()`, atomic train/test splitting, and conversion to matrices for model input
  * Pipeline — bundles a `StandardScaler` and a model into one value type, so the scaler trained on the training set is the exact scaler applied at predict time
  * Feature scaling — `StandardScaler` for z-score scaling (default for distance-based models) and `FeatureScaler` for min-max scaling, both with fit-then-transform
  * Train/test split with reproducible seeded shuffling
  * K-fold cross-validation — `kFoldIndices(k:seed:)` returns leak-free train/validation index pairs so a tuning choice is judged on data the model did not fit
  * Stratified splitting preserving class proportions
  * Boolean masking and conditional selection
  * Class distribution, imbalance ratio, and oversampling for imbalanced data

* **Data Visualization**
  * Trend lines via linear regression predict
  * Elbow curves for K-Means cluster selection
  * Confusion matrix heatmap data
  * Softmax probability bars
  * Downsampling, stacked series, correlation matrices

* **Machine Learning Models**
  * Gaussian Naive Bayes with calibrated `predictProbabilities` for soft-output classification
  * K-Nearest Neighbors (classification with Euclidean and cosine distance)
  * Linear Regression with single-feature convenience predict
  * Gradient Descent — fits a regression by iterative optimization, exposing the full `lossHistory` and a typed `Outcome` so convergence is observable rather than hidden
  * Ridge Regression — L2-regularized regression that curbs overfitting and absorbs collinear-feature instability, fit by the same descent optimizer
  * K-Means Clustering with elbow method and multi-seed best fit
  * Consistent `fit`/`predict` API across all models
  * Full transparency — inspect coefficients, centroids, priors, means, variances
  * All models conform to `Codable` — train once, encode to JSON, decode on any platform

* **Evaluation Metrics**
  * Confusion matrix with accuracy, precision, recall, F1 score
  * Per-class classification report with macro and weighted averages
  * R², MSE, RMSE for regression
  * All metrics available programmatically as typed values

* **Activation Functions**
  * Softmax (multi-class probability distribution)
  * Sigmoid (binary classification threshold)

* **Text and Embeddings**
  * Tokenization for NLP workflows
  * Word embedding lookups with vocabulary filtering

## Quick Start

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

## Quiver Notebook

[Quiver Notebook](https://github.com/waynewbishop/quiver-notebook) is a browser-based Swift editor with Quiver and a library of teaching datasets pre-loaded. Write Swift snippets against bundled datasets like iris, Titanic, California Housing, and student-performance — no Xcode project, no package setup, no CSV hunting. Snippets carry over unchanged into an iOS, watchOS, or Vapor app, so the same code that taught the concept ships the feature. See the [Quiver Notebook README](https://github.com/waynewbishop/quiver-notebook) for setup.

## Design Philosophy

* **Swift-first**: Extends standard Swift arrays — no custom container types, no conversion overhead
* **Transparent ML**: Every model exposes its learned parameters — inspect coefficients, centroids, and class statistics
* **Educational**: Clear implementations that map directly to mathematical concepts
* **Zero dependencies**: Pure Swift, no external frameworks required

## When to Use Quiver

* **Education** — Teach vector math, statistics, and ML concepts with readable, inspectable code
* **Machine learning** — Train and evaluate classification, regression, and clustering models in pure Swift with full algorithm transparency
* **On-device intelligence** — Privacy-first ML on iOS, watchOS, and visionOS without cloud dependencies or external runtimes
* **Data visualization** — Prepare chart-ready arrays (trend lines, elbow curves, heatmaps) for Swift Charts
* **Server-side Swift** — Run numerical computing and ML pipelines on Linux and Vapor

## Cookbook

[59 interactive recipes](https://github.com/waynewbishop/quiver-cookbook) for learning vector math, statistics, and ML models in Swift. Each recipe is a single `.swift` file optimized for the Xcode `#Playground` macro — clone the repo, open in Xcode, and start experimenting.

## Documentation

Full API documentation at [waynewbishop.github.io/quiver](https://waynewbishop.github.io/quiver/documentation/quiver/), including:
* Vector operations and linear algebra primer
* Statistical function reference
* Matrix operations and transformations
* Data preparation and visualization guides
* Machine learning model guides (Naive Bayes, K-Nearest Neighbors, Linear Regression, Gradient Descent, Ridge Regression, K-Means)

Quiver is the companion numerical computing package for [Swift Algorithms & Data Structures](https://waynewbishop.github.io/swift-algorithms/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Quiver is available under the Apache License, Version 2.0. See the LICENSE file for more info.

## Newsletter

[The Feature Vector](https://featurevector.substack.com) — a newsletter about ML intuition for engineers, built in Swift. One idea per issue, with a recipe you can run in Xcode.

## Questions

Have a question? Feel free to contact me on [LinkedIn](https://www.linkedin.com/in/waynebishop).
