# Data Visualization

Prepare, scale, and reshape data for Swift Charts and other visualization frameworks.

## Overview

Quiver provides a set of functions that bridge numerical data and chart-ready output. These operations handle the data preparation step — scaling values to a target range, computing frequency distributions, stacking series for area charts, and downsampling large datasets for responsive rendering. Each function returns structured output that maps directly to Swift Charts mark types.

### Scaling and normalization

Raw data often needs scaling before visualization. Quiver offers three approaches depending on the use case:

```swift
import Quiver

let revenues = [45000.0, 52000.0, 48000.0, 61000.0, 55000.0]

// Min-max scaling to a custom range (e.g., for sizing chart elements)
let sizes = revenues.scaled(to: 10.0...50.0)
// [10.0, 27.5, 17.5, 50.0, 35.0]

// Z-score standardization (mean=0, std=1) for comparing distributions
let standardized = revenues.standardized()
// [-1.29, -0.04, -0.75, 1.58, 0.50]

// Convert to percentages of total (for pie/donut charts)
let shares = revenues.asPercentages()
// [17.2, 19.9, 18.4, 23.4, 21.1]
```

Min-max scaling with `scaled(to:)` maps values to any closed range, which is useful for controlling the visual size of chart marks. Standardization with `standardized` centers data around zero, making it possible to overlay series with different units on the same axis. Percentages with `asPercentages` express each value as a share of the total, ready for proportional charts.

### Stacked series

Stacked area and bar charts require cumulative data where each series builds on the one below it. Quiver handles this transformation for both absolute and percentage-based stacking:

```swift
import Quiver

let mobile  = [120.0, 135.0, 150.0, 140.0]
let desktop = [200.0, 190.0, 210.0, 195.0]
let tablet  = [50.0, 55.0, 45.0, 60.0]

let series = [mobile, desktop, tablet]

// Cumulative stacking (each series adds to the previous)
let stacked = series.stackedCumulative()
// stacked[0] = [120.0, 135.0, 150.0, 140.0]     (mobile)
// stacked[1] = [320.0, 325.0, 360.0, 335.0]     (mobile + desktop)
// stacked[2] = [370.0, 380.0, 405.0, 395.0]     (all three)

// Percentage stacking (each time point sums to 100%)
let percents = series.stackedPercentage()
// percents[0] = [32.4, 35.5, 37.0, 35.4]   (mobile %)
// percents[1] = [54.1, 50.0, 51.9, 49.4]   (desktop %)
// percents[2] = [13.5, 14.5, 11.1, 15.2]   (tablet %)
```

### Correlation heatmaps

Visualize relationships between multiple variables using a correlation matrix flattened into chart-ready tuples:

```swift
import Quiver

let temperature = [30.0, 32.0, 35.0, 28.0, 33.0]
let iceCream    = [200.0, 220.0, 260.0, 180.0, 230.0]
let hotCocoa    = [150.0, 130.0, 100.0, 170.0, 120.0]

// Compute the correlation matrix
let matrix = [temperature, iceCream, hotCocoa].correlationMatrix()
// [[1.0,   0.99, -0.99],
//  [0.99,  1.0,  -0.98],
//  [-0.99, -0.98, 1.0]]

// Flatten to (x, y, value) tuples for heatmap rendering
let labels = ["Temp", "Ice Cream", "Hot Cocoa"]
let heatmap = [temperature, iceCream, hotCocoa].heatmapData(labels: labels)
// [("Temp", "Temp", 1.0), ("Temp", "Ice Cream", 0.99), ...]
```

Each tuple maps directly to a `RectangleMark` in Swift Charts, with the value driving color intensity.

### Downsampling

Large datasets can slow chart rendering. Downsampling reduces data volume while preserving the shape of the signal by aggregating values within fixed-size windows:

```swift
import Quiver

// Hourly readings over a day
let hourlyTemps = [
    18.0, 17.5, 17.0, 16.5, 16.0, 16.5,
    18.0, 20.0, 22.0, 24.0, 25.5, 26.0,
    27.0, 27.5, 27.0, 26.0, 24.5, 23.0,
    21.0, 19.5, 18.5, 18.0, 17.5, 17.0
]

// Reduce to 4 six-hour averages
let sixHourly = hourlyTemps.downsample(factor: 6, using: .mean)
// [16.9, 22.6, 25.8, 18.6]

// Keep the peak value in each window instead
let sixHourlyMax = hourlyTemps.downsample(factor: 6, using: .max)
// [18.0, 26.0, 27.5, 21.0]
```

The `AggregationMethod` parameter controls how values within each window are combined: `.mean` for smoothed trends, `.max` or `.min` for extremes, `.sum` for totals, `.count` for frequency, and `.percentage` for group sums normalized to 100%.

### Time series smoothing and differentiation

Time series data — sensor readings, financial prices, health metrics — often needs smoothing or rate-of-change computation before visualization.

The `rollingMean(window:)` method computes a simple moving average where every point in the window carries equal weight. For signals where recent values matter more, `exponentialMean(alpha:)` gives exponentially decreasing weight to older values, making it more responsive to recent changes:

```swift
import Quiver

let heartRate = [142.0, 145.0, 155.0, 148.0, 150.0, 162.0, 158.0]

// Simple moving average — equal weight across the window
let simple = heartRate.rollingMean(window: 3)
// [142.0, 143.5, 147.3, 149.3, 151.0, 153.3, 156.7]

// Exponential moving average — recent values weighted more heavily
let smoothed = heartRate.exponentialMean(alpha: 0.3)
// [142.0, 142.9, 146.5, 147.0, 147.9, 152.1, 153.9]

// Span-based convenience (matches common financial conventions)
let ema5 = heartRate.exponentialMean(span: 5)
```

The `derivative(sampleRate:)` method computes the rate of change between consecutive measurements. This converts position to velocity, velocity to acceleration, or any signal to its instantaneous rate of change:

```swift
import Quiver

// Elevation samples at 1-second intervals
let elevation = [100.0, 102.0, 105.0, 104.0, 107.0]
let grade = elevation.derivative(sampleRate: 1.0)
// [2.0, 3.0, -1.0, 3.0] — meters of climb per second

// Speed samples at 0.5-second intervals
let speed = [3.0, 3.5, 4.2, 4.0]
let acceleration = speed.derivative(sampleRate: 0.5)
// [1.0, 1.4, -0.4] — acceleration in m/s²
```

The result has one fewer element than the input because each derivative requires two adjacent values. The `sampleRate` parameter represents the time between consecutive measurements — dividing the raw difference by this interval produces the correct physical units.

### Filtering with boolean masks

Boolean masks split a dataset into separate series based on conditions. This is useful for separating outliers from normal readings, highlighting predictions above a confidence threshold, or splitting data into labeled groups before charting:

```swift
import Quiver

let readings = [23.5, 24.1, 150.0, 23.8, 22.9, -10.0, 24.5]
let timestamps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

// Split into valid readings and outliers
let valid = readings.isGreaterThanOrEqual(0.0)
    .and(readings.isLessThanOrEqual(50.0))

let normalReadings = readings.masked(by: valid)
let normalTimes = timestamps.masked(by: valid)
// normalReadings: [23.5, 24.1, 23.8, 22.9, 24.5]
// normalTimes:    [1.0, 2.0, 4.0, 5.0, 7.0]

let outlierReadings = readings.masked(by: valid.not)
let outlierTimes = timestamps.masked(by: valid.not)
// outlierReadings: [150.0, -10.0]
// outlierTimes:    [3.0, 6.0]
```

The two arrays — normal and outlier — are parallel arrays ready for separate chart series. Normal readings can render as one color and outliers as another, making anomalies visually distinct without any manual filtering logic.

> Note: For more on boolean comparisons, logical operators, and masking, see <doc:Boolean-Masking>.

### Visualizing machine learning results

Quiver's ML models produce outputs that map directly to Swift Charts marks.

**Confusion matrix heatmap.** A binary classifier's `ConfusionMatrix` contains four counts — true positives, false positives, true negatives, and false negatives. These map to a 2×2 `RectangleMark` heatmap using `heatmapData`:

```swift
import Quiver

let predictions = model.predict(scaler.transform(testX))
let cm = predictions.confusionMatrix(actual: testY)

// Build the 2×2 grid as a matrix
let grid: [[Double]] = [
    [Double(cm.trueNegatives), Double(cm.falsePositives)],
    [Double(cm.falseNegatives), Double(cm.truePositives)]
]

// Flatten to (x, y, value) tuples for RectangleMark
let labels = ["Predicted 0", "Predicted 1"]
let heatmap = grid.heatmapData(labels: labels)
```

Each tuple in `heatmap` carries the row label, column label, and count — ready for a `RectangleMark` with color intensity driven by the value. This visualization makes it immediately clear where the model confuses one class for another.

**Elbow method for K-Means.** Choosing the right number of clusters is the central question in K-Means. The elbow method runs the algorithm for several values of `k` and plots inertia (total within-cluster distance) against `k`. The "elbow" — where inertia stops dropping sharply — suggests the natural number of clusters:

```swift
import Quiver

let data: [[Double]] = // ... feature matrix

// Compute inertia for k = 1 through 8
let kRange = Array(1...8)
let inertias = KMeans.elbowMethod(data: data, kRange: kRange, seed: 42)

// kRange and inertias are parallel arrays — plot as LineMark
// x: kRange[i], y: inertias[i]
```

The result is a line chart where each point is one `LineMark`. A sharp bend in the curve indicates the point where adding more clusters stops providing meaningful improvement.

**Regression line overlay.** Linear regression produces coefficients that define a line (or hyperplane — the same idea extended to more than two dimensions). For single-feature regression, we can overlay the fitted line on a scatter plot of the original data:

```swift
import Quiver

let model = try LinearRegression.fit(features: trainX, targets: trainY)

// Generate prediction line across the feature range
let xValues = [Double].linspace(start: 0.0, end: 10.0, count: 50)
let yValues = model.predict(xValues)

// trainX/trainY — scatter points
// xValues/yValues — fitted trend line (50 evenly spaced points)
```

The scatter shows the raw data, and the line shows what the model learned. The gap between points and line is the residual error — visible at a glance.

**Polynomial trend line.** When the relationship between `x` and `y` is curved rather than linear, `polyfit(x:y:degree:)` returns a `Polynomial` that captures the curve. Evaluating the polynomial across an evenly spaced grid produces a smooth overlay for Swift Charts, the same shape as the linear regression overlay above but with a quadratic (or higher-degree) fit:

```swift
import Quiver

// Curved data — points roughly follow y = 2x² + 3x + 1
let x = [1.0, 2.0, 3.0, 4.0, 5.0]
let y = [6.5, 14.8, 28.1, 44.9, 66.2]

if let p = [Double].polyfit(x: x, y: y, degree: 2) {
    // Evaluate the curve across a dense grid for plotting
    let xValues = Array.linspace(start: 0.5, end: 5.5, count: 50)
    let yValues = p(xValues)

    // x/y     — scatter points
    // xValues/yValues — fitted polynomial trend (50 evenly spaced points)
}
```

The same overlay pattern works for any polynomial degree. Higher degrees fit more flexible curves at the cost of overfitting risk on small samples — the right degree is the one that captures the trend without bending around noise.

**SoftMax probability distribution.** The `softMax` function converts raw model scores into a probability distribution that sums to 1.0. This maps naturally to a `BarMark` showing confidence per class:

```swift
import Quiver

// Raw model scores (logits) — higher means more confident
let scores = [2.1, 0.8, -0.3, 1.5]
let probs = scores.softMax()
// [0.52, 0.14, 0.05, 0.29]

let classNames = ["cat", "dog", "bird", "fish"]
// Plot classNames[i] vs probs[i] as BarMark
```

The tallest bar is the model's prediction. The relative heights show how confident the model is — a single dominant bar means high confidence, while similar heights across bars suggest uncertainty.

## Topics

### Scaling and normalization
- ``Swift/Array/scaled(to:)``
- ``Swift/Array/standardized()``
- ``Swift/Array/asPercentages()``

### Distribution analysis
- ``Swift/Array/histogram(bins:)``
- ``Swift/Array/quartiles()``
- ``Swift/Array/percentile(_:)``
- ``Swift/Array/percentileRank(of:)``
- ``Swift/Array/percentileRanks()``

### Multi-series operations
- ``Swift/Array/stackedCumulative()``
- ``Swift/Array/stackedPercentage()``
- ``Swift/Array/correlationMatrix()``
- ``Swift/Array/heatmapData(labels:)``

### Downsampling
- ``Swift/Array/downsample(factor:using:)``

### Time series
- ``Swift/Array/rollingMean(window:)``
- ``Swift/Array/exponentialMean(alpha:)``
- ``Swift/Array/exponentialMean(span:)``
- ``Swift/Array/derivative(sampleRate:)``
- ``Swift/Array/diff(lag:)``
- ``Swift/Array/percentChange(lag:)``

### Rounding
- ``Swift/Array/rounded(to:)``

### Grouping and aggregation
- ``Swift/Array/groupBy(_:using:)``
- ``Swift/Array/groupedData(by:using:)``
