# Boolean Masking

Filter and select array elements using comparison operators, logical conditions, and boolean masks.

## Overview

Boolean masking is how Quiver filters data before it reaches a model. Raw datasets contain outliers, missing values, and irrelevant samples that degrade predictions. Quiver's comparison operators produce boolean arrays that act as masks — selecting the elements that meet a condition and discarding the rest. This is the same pattern used by `Panel.filtered(where:)` to clean feature columns before training.

### Comparisons

Compare array elements against a threshold to produce a boolean mask:

```swift
import Quiver

let predictions = [0.92, 0.45, 0.87, 0.31, 0.78]

// Which predictions exceed the confidence threshold?
let confident = predictions.isGreaterThanOrEqual(0.75)
// [true, false, true, false, true]
```

Quiver provides a full set of comparison operators that work element-wise on arrays:

```swift
let scores = [85.0, 45.0, 92.0, 38.0, 76.0]

scores.isGreaterThan(80.0)          // [true, false, true, false, false]
scores.isLessThan(50.0)            // [false, true, false, true, false]
scores.isGreaterThanOrEqual(76.0)  // [true, false, true, false, true]
scores.isLessThanOrEqual(45.0)     // [false, true, false, true, false]
```

Two arrays can also be compared element-by-element. This is useful for checking predictions against expected values:

```swift
let predicted = [92.0, 88.0, 95.0, 78.0]
let actual = [90.0, 90.0, 95.0, 85.0]

let matches = predicted.isEqual(to: actual)
// [false, false, true, false]
```

### Combining conditions

Boolean arrays support `.and()`, `.or()`, and `.not` for building compound filters. In data preparation, this is how we apply multiple constraints simultaneously — for example, keeping only the samples where features fall within a valid range:

```swift
import Quiver

let features = [23.5, 24.1, 150.0, 23.8, 22.9, -10.0, 24.5]

// Keep readings within a valid range (0-50)
let inRange = features.isGreaterThanOrEqual(0.0)
    .and(features.isLessThanOrEqual(50.0))
// [true, true, false, true, true, false, true]

// Invert to find the outliers
let outliers = inRange.not
// [false, false, true, false, false, true, false]
```

The `.or()` operator selects elements that meet either condition:

```swift
let temperatures = [15.0, 22.0, 35.0, 18.0, 28.0]
let humidity = [60.0, 80.0, 45.0, 70.0, 65.0]

// Flag samples where temperature OR humidity is extreme
let tempExtreme = temperatures.isGreaterThan(30.0)
    .or(temperatures.isLessThan(10.0))
let humidExtreme = humidity.isGreaterThan(75.0)

let flagged = tempExtreme.or(humidExtreme)
// [false, true, true, false, false]
```

### Applying masks

Once a boolean mask identifies which elements to keep, `masked(by:)` extracts them and `trueIndices` returns their positions:

```swift
import Quiver

let features = [23.5, 24.1, 150.0, 23.8, 22.9, -10.0, 24.5]

// Filter to valid range
let valid = features.isGreaterThanOrEqual(0.0)
    .and(features.isLessThanOrEqual(50.0))

// Extract clean data for training
let cleanFeatures = features.masked(by: valid)
// [23.5, 24.1, 23.8, 22.9, 24.5]

// Find which indices were kept
let keptIndices = valid.trueIndices
// [0, 1, 3, 4, 6]

// Extract values with their original positions preserved
let flagged = features.maskedWithIndices(by: valid.not)
// [(index: 2, value: 150.0), (index: 5, value: -10.0)]
```

The `maskedWithIndices(by:)` method is useful when we need to know *which* elements matched — for example, annotating outlier points on a chart with their day number or labeling flagged values in a report.

This integrates directly with `Panel`. When filtering a `Panel`, the same mask applies to every column simultaneously, keeping rows aligned across all features:

```swift
import Quiver

let data = Panel([
    ("age", [25.0, 30.0, 200.0, 28.0]),
    ("income", [50000.0, 60000.0, 75000.0, 55000.0])
])

// Remove rows with invalid age
let validAge = data["age"].isLessThanOrEqual(120.0)
let cleaned = data.filtered(where: validAge)
// cleaned.rowCount == 3
```

### Conditional selection

`choose(where:otherwise:)` selects elements from one array where a condition is true, and from another array where it is false. This is useful for replacing outliers or imputing default values before training:

```swift
import Quiver

let readings = [23.5, 150.0, 22.9, -10.0, 24.5]
let valid = readings.isGreaterThanOrEqual(0.0)
    .and(readings.isLessThanOrEqual(50.0))

// Replace invalid readings with a default value
let defaults = [Double](repeating: 25.0, count: readings.count)
let cleaned = readings.choose(where: valid, otherwise: defaults)
// [23.5, 25.0, 22.9, 25.0, 24.5]
```

## Topics

### Comparisons
- ``Swift/Array/isEqual(to:)``
- ``Swift/Array/isGreaterThan(_:)``
- ``Swift/Array/isLessThan(_:)``
- ``Swift/Array/isGreaterThanOrEqual(_:)``
- ``Swift/Array/isLessThanOrEqual(_:)``

### Boolean logic
- ``Swift/Array/and(_:)``
- ``Swift/Array/or(_:)``
- ``Swift/Array/not``

### Masking and filtering
- ``Swift/Array/masked(by:)``
- ``Swift/Array/maskedWithIndices(by:)``
- ``Swift/Array/choose(where:otherwise:)``
- ``Swift/Array/trueIndices``

### See also
- <doc:Panel>
- <doc:Feature-Scaling>
