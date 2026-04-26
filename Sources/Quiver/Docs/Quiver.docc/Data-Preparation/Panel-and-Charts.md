# Panel and Charts

Aggregating data with Panel and feeding it directly into Swift Charts.

## Overview

Swift Charts is excellent at rendering. It is not opinionated about where the data comes from or how it has been summarized — by the time a `BarMark` or `LineMark` is constructed, the values already need to be in the right shape. iOS developers building data-driven views routinely have the opposite problem: a `[Double]` of raw measurements from a sensor, an analytics endpoint, or a CSV import, with no aggregation step yet.

Panel is the bridge. It organizes raw columns into named features, supports filtering and splitting across all columns at once, and converts to the matrix and scalar shapes that the rest of Quiver — and Swift Charts — expects. As Wayne has put it: "One of the goals for Quiver was to provide a data aggregation model for easy Swift Charts integration." This article walks that integration end to end, from raw arrays to a chart on screen.

### What Panel provides

Panel is a value type that holds named columns of `[Double]` data with rows aligned across columns. The full reference for the type, including initializers, splitting, and the design rationale, lives in <doc:Panel>. The pieces that matter for charting are a focused subset: `head(n:)` for previewing rows in tabular form, `summary()` for per-column statistics, `filtered(where:)` for boolean-mask filtering across all columns at once, and `toMatrix(columns:)` for extracting selected columns as `[[Double]]` when a chart needs to iterate rows.

Each column returned by subscript is a plain `[Double]`, which means every Quiver vector operation — `mean()`, `std()`, `groupedData(by:using:)`, `rollingMean(window:)` — applies directly. That is the connective tissue between Panel and the rest of the data-preparation surface in <doc:Data-Visualization>.

### From raw data to Panel

A Panel begins as ordered named columns of equal length. The columns can come from anywhere — a JSON decode, a CSV parse, a series of HealthKit samples — but the contract is always the same: each column is a `[Double]`, and every column has the same row count:

```swift
import Quiver

let workouts = Panel([
    ("week",      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ("heartRate", [142.0, 148.0, 151.0, 145.0, 153.0, 149.0]),
    ("cadence",   [168.0, 172.0, 175.0, 170.0, 178.0, 174.0])
])

print(workouts.head(n: 3))
//    week  heartRate  cadence
// 0   1.0      142.0    168.0
// 1   2.0      148.0    172.0
// 2   3.0      151.0    175.0

print(workouts.summary())
// One row per column: count, mean, std, min, max
```

The `head()` output shows what is actually in the panel after construction. The `summary()` output gives the per-column descriptive statistics — mean, standard deviation, minimum, maximum — that often answer the question being asked before any chart is needed at all.

### Plotting Panel data with Swift Charts

Once a panel is in hand, the path to a chart is short. Swift Charts iterates data and emits one mark per row, so the chart-side code asks for two things: an iterable collection and a stable identifier per element. Panel provides both — the row count is known, and each column can be read as a parallel `[Double]`:

```swift
import Charts
import Quiver
import SwiftUI

struct WorkoutTrendChart: View {
    let workouts: Panel

    var body: some View {
        Chart {
            ForEach(0..<workouts.rowCount, id: \.self) { row in
                LineMark(
                    x: .value("Week", workouts["week"][row]),
                    y: .value("Heart Rate", workouts["heartRate"][row])
                )
            }
        }
    }
}
```

The `Chart { ForEach { LineMark } }` pattern is the canonical Swift Charts shape for time-series data, and the columns of a Panel slot in directly. For categorical aggregations — total revenue per region, mean response time per endpoint, count of events per day — the natural starting point is `groupedData(by:using:)` on a `[Double]` column, which returns sorted `(category, value)` tuples that map straight to a `BarMark`:

```swift
let sales: [Double]   = [120.0, 95.0, 140.0, 110.0, 85.0, 130.0]
let regions: [String] = ["North", "South", "North", "South", "South", "North"]

let chartData = sales.groupedData(by: regions, using: .sum)
// [(category: "North", value: 390.0), (category: "South", value: 290.0)]

// In a Swift Charts view:
// Chart(chartData, id: \.category) { item in
//     BarMark(x: .value("Region", item.category),
//             y: .value("Total",  item.value))
// }
```

The grouping happens once, in Quiver. The chart receives sorted, labeled tuples and renders them. There is no second pass to deduplicate categories, no `Dictionary` to flatten on the chart side.

### Choosing the chart shape

The aggregation step usually decides the chart shape. A categorical sum or mean — `groupedData(by:using: .sum)`, `.mean`, `.count` — is a `BarMark`. A time series, optionally smoothed with `rollingMean(window:)` or `exponentialMean(alpha:)`, is a `LineMark`. A distribution summarized with `histogram(bins:)` is a `BarMark` over bin midpoints. A correlation surface produced by `heatmapData(labels:)` is a `RectangleMark` driven by the `value` field of each tuple.

The full catalog of chart-ready transformations — stacked series, percentile ranks, scaled-to-range outputs, downsampled signals — is documented in <doc:Data-Visualization>. The pattern is consistent across all of them: the Quiver method returns a shape that matches a Swift Charts mark, and the chart code stays small.

## See also

- <doc:Panel>
- <doc:Data-Visualization>
- <doc:Statistical-Operations>
