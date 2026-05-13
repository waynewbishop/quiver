# iOS Patterns

Patterns that turn arrays an iOS app already holds into charts and inferential answers.

## Overview

A finance app wants to render a year of transactions as a donut chart by category. A health app wants to fold thirty days of readings into a weekly trend. A feed wants to highlight the unusual entries without burying the normal ones. A workout app wants to tell the user whether this week's pace is genuinely faster than last week's or just noise. Each situation has the same shape: raw values on one side, an answer on the other, and a clean piece of math in between. Quiver gives us the building blocks to compose that answer from the arrays the app already holds. This article walks through four patterns that come up repeatedly.

### Aggregating data for Swift Charts

[Swift Charts](https://developer.apple.com/documentation/charts) is excellent at rendering. It is not particular about where the data comes from or how it has been summarized — by the time a `BarMark`, `SectorMark`, or `PointMark` is constructed, the values already need to be in the right shape. The Quiver side of the pipeline gets the values into that shape. The chart side reads them and renders.

The three chart types most iOS dashboards reach for each pair with one Quiver method that does the aggregation work:

```swift
import Charts
import Quiver
import SwiftUI

struct FinanceDashboard: View {
    let amounts: [Double]
    let categories: [String]

    // Sums the spending in each category and returns sorted percentage shares
    private var categoryShares: [(category: String, value: Double)] {
        amounts.groupedData(by: categories, using: .percentage)
    }

    var body: some View {
        Chart(categoryShares, id: \.category) { share in
            SectorMark(
                angle: .value("Share", share.value),
                innerRadius: .ratio(0.6)
            )
            .foregroundStyle(by: .value("Category", share.category))
        }
    }
}
```

The `groupedData(by:using:)` call is the load-bearing step. It accepts a parallel array of category labels and an aggregation kind — `.sum`, `.mean`, `.count`, `.percentage` — and returns sorted `(category, value)` tuples that map directly to the shape `SectorMark` (donut) or `BarMark` (bar) expects. No `Dictionary` to flatten on the chart side, no second pass to deduplicate categories.

For headline statistics that sit above the chart — total spending, daily average, month-over-month change — the same pattern applies, just with vector aggregation:

```swift
let total = amounts.sum()                          // 5050.0
let dailyAverage = amounts.mean() ?? 0             // 32.99
let change = amounts.percentChange(lag: 1)         // [-0.04, 0.02, ...]
```

Each call returns a `Double` or `[Double]` that drops straight into a SwiftUI `Text` view. There is no intermediate container, no formatter layer, no view-model state that the chart and the headline statistics have to share.

### Downsampling a long time series

Time series data on iOS often arrives at a finer cadence than the screen needs. Daily transactions become a weekly bar chart. Hourly readings become a daily summary. Per-second samples become a per-minute trace. The `downsample(factor:using:)` method collapses every N adjacent values into a single value, with the same aggregation kinds as `groupedData`.

```swift
import Charts
import Quiver

struct WeeklyBreakdown: View {
    let dailySpending: [Double]   // 30 values, one per day

    // Collapse 30 days into 5 weekly totals
    private var weeklyTotals: [Double] {
        dailySpending.downsample(factor: 6, using: .sum)
    }

    var body: some View {
        Chart {
            ForEach(Array(weeklyTotals.enumerated()), id: \.offset) { index, value in
                BarMark(
                    x: .value("Week", index + 1),
                    y: .value("Spending", value)
                )
            }
        }
    }
}
```

The downsample is a single array transform. The chart receives a small, view-sized array and renders without ever needing to know there was a longer signal behind it. For finer-grained smoothing where the cadence shouldn't change — flattening noise without losing samples — see `rollingMean(window:)` and `exponentialMean(alpha:)` in <doc:Data-Visualization>.

For outlier-aware time series — where the goal is to render every sample but highlight the unusual ones — the pattern combines `outlierMask(threshold:)` with `maskedWithIndices(by:)` so the chart can render normal values one way and outliers another:

```swift
import Charts
import Quiver

struct UnusualDaysChart: View {
    let dailySpending: [Double]

    private var outlierFlags: [Bool] {
        dailySpending.outlierMask(threshold: 1.5)
    }

    var body: some View {
        Chart {
            ForEach(Array(dailySpending.enumerated()), id: \.offset) { day, value in
                PointMark(
                    x: .value("Day", day + 1),
                    y: .value("Spending", value)
                )
                .foregroundStyle(outlierFlags[day] ? .red : .blue)
                .symbolSize(outlierFlags[day] ? 120 : 40)
            }
        }
    }
}
```

The mask is computed once per render against the current data. SwiftUI re-runs the body only when the input changes, so the work is paid for at the moments the dashboard refreshes — not on every frame.

### Decoding sensor and storage inputs

iOS apps have many places to read numbers from — `HKHealthStore` for health data, `CMMotionManager` for motion, `UserDefaults` and `FileManager` for what the app has stored itself, the keyboard and form fields for user input. Quiver does not interact with any of these sources directly. The pattern is to decode whatever the source produces into `[Double]` once, near the boundary, and treat the result as Quiver data from that point on.

```swift
import Quiver

// Once HealthKit samples are in a [Double], a Panel turns them into a typed summary
func summarizeHeartRate(samples: [Double]) -> PanelSummary? {
    let panel = Panel([("heartRate", samples)])
    return panel.summary()
}
```

Calling `Panel.summary()` returns a `PanelSummary` with count, mean, standard deviation, quartiles, min, max, and IQR — every field a dashboard typically needs in one typed value. The result is `Codable` and `Sendable`, so it crosses task boundaries and persists to disk without ceremony. Each column's statistics live in a `ColumnSummary` value addressable by field name. See <doc:Panel-Workflows> for the full typed-summary surface.

The source — an `HKQuantitySample` query callback, an array of `CMAccelerometerData` readings, a CSV column the user imported, a list of numeric form entries — becomes a `[Double]` once at the boundary. Everything downstream is the same Quiver code that runs on any other input. Decode at the edge, compute in the middle, render at the end.

> Tip: When the dashboard needs more than a summary — multiple aligned columns, filtering across rows, splitting into train/test subsets, or charting — the full `Panel` surface picks up where `summary()` leaves off. See <doc:Panel> for the type itself and <doc:Panel-Workflows> for the split, summary, classification, and charting patterns.

### Turning a reading into a percentile

A health dashboard records heart-rate readings over months and wants to tell the user how unusual today's resting reading is. The user's history makes that possible. The mean and standard deviation of past readings define a personal distribution; `Distributions.normal.cdf` turns the current reading into the probability of seeing a value at least that extreme, which the UI can render as "in the top 2% of your history" or similar.

```swift
import Quiver

// User's historical resting heart rates over the past three months
let history: [Double] = [/* … */]
let today: Double = 84

if let mean = history.mean(),
   let std = history.standardDeviation() {
    let z = (today - mean) / std
    if let cdf = Distributions.normal.cdf(x: z, mean: 0, standardDeviation: 1) {
        let upperTail = 1.0 - cdf
        let percentLabel = String(format: "Top %.0f%% of your history", upperTail * 100)
        // Render percentLabel in a glance complication, a SwiftUI Text, or a notification body
    }
}
```

The whole calculation runs on the watch or phone with no network call and no permissions beyond what the app already had to read history. A glance can render "this reading is in the top 2% of your history" the moment a sample arrives.

### Small-sample A/B comparisons inside the app

iOS apps that surface their own analytics often have a sample size of ten or twenty observations, not thousands. A workout app showing the user "is your pace this week genuinely faster than last week, or is the gap just noise?" needs a t-test, not a normal z-test, because the sample is small. Calling `Distributions.t.cdf` produces the honest p-value at small `df`.

```swift
import Quiver

let thisWeek: [Double] = [/* 8 pace samples */]
let baseline: Double = 9.5  // user's typical pace from a longer baseline

if let mean = thisWeek.mean(),
   let se = thisWeek.standardError() {
    let t = (mean - baseline) / se
    let df = Double(thisWeek.count - 1)
    if let cdf = Distributions.t.cdf(x: abs(t), df: df) {
        let pValue = 2 * (1 - cdf)
        let label = pValue < 0.05
            ? "Significantly different from your typical pace"
            : "Within your typical range"
    }
}
```

> Note: The same calculation against the normal distribution would overstate significance at small `n`. For ten samples the t-distribution gives a p-value roughly five times larger than the normal would, which is the honest accounting of small-sample uncertainty. For the full distribution surface, see <doc:Working-With-Distributions>.

> Experiment: [quiver-demo-ios](https://github.com/waynewbishop/quiver-demo-ios) is a personal-finance dashboard that renders three Swift Charts views from twenty-four hardcoded transactions. Cloning it and changing the `outlierMask` threshold in `FinanceModel.swift` from `1.5` to `1.0` and then to `2.5` shows the scatter's red points expand and contract — the clearest way to feel what a z-score threshold controls.
