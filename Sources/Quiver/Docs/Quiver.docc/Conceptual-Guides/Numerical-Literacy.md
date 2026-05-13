# Numerical Literacy

Reason about floating-point precision, accumulated error, and the trustworthiness of numerical results.

## Overview

Every number in Quiver is a `Double` — a 64-bit floating-point value that approximates a real number using a finite pattern of bits. Most of the time, that approximation is close enough that we can ignore it. The mean of a column, the dot product of two vectors, the determinant of a matrix — these compute to results that match what we would get with pencil and paper to as many decimal places as anyone reasonably needs.

But the approximation is still an approximation. **Numerical literacy** is the skill of knowing when the gap between a floating-point result and the true mathematical answer matters, and what to do about it when it does. This primer is about that skill: how floating-point numbers behave, where Quiver protects us by default, the contract Quiver uses to flag a result the math could not produce, and the warning signs that tell us to look closer.

> Note: This primer pairs with the <doc:Statistics-Primer> and <doc:Determinants-Primer>. Many of the examples here are concrete instances of the abstractions described there.

### What a Double represents

A `Double` has 64 bits, divided into a sign, an exponent, and a 52-bit mantissa. The mantissa holds the significant digits of the number while the exponent says where the decimal point sits, much like scientific notation. That mantissa gives roughly 15 to 17 significant decimal digits of precision. For most data — physical measurements, money in dollars, ratios, percentages — that is far more precision than the input itself carries.

The catch is that `Double` represents numbers in binary, not decimal. Some decimal values that look exact on paper have no exact binary representation. The classic example:

```swift
import Quiver

let a = 0.1
let b = 0.2
let sum = a + b
// sum is 0.30000000000000004, not 0.3
```

Neither `0.1` nor `0.2` can be written exactly in binary. They round to the nearest representable value, and the rounding errors compound when we add them. The result is correct to about 16 significant digits, which is the best a `Double` can do.

This rounding behavior is the source of most floating-point surprises. A test that asserts `a + b == 0.3` will fail. A computation that subtracts two nearly equal numbers can lose most of its precision in a single step. A loop that accumulates millions of small values can drift from the true answer.

> Tip: Compare floating-point values with a tolerance — `abs(a - b) < 1e-9` is a reliable default. The `1e-9` is Swift's shorthand for `0.000000001` — the `e` reads as "times ten to the," so `1e-9` is one billionth. Quiver follows this rule internally and we follow it in any code that consumes Quiver's output.

### Where Quiver protects us

The most common way for floating-point error to dominate a result is by subtracting two numbers that are nearly equal. Each input has 15–17 significant digits, but when we subtract them the leading digits cancel and only the noisy trailing digits remain. The textbook formula for variance — `mean(x²) − mean(x)²` — falls into this trap whenever the mean is large compared to the spread of the data. Quiver's `variance` and `standardDeviation` use a different formulation that subtracts the mean before squaring, avoiding the cancellation entirely. Calling `variance` on a tightly clustered dataset returns the right answer; rolling our own from the textbook formula often does not. See <doc:Statistics-Primer> for the math behind these summaries.

The same care extends to iterative algorithms. The `KMeans` model recomputes cluster centroids by averaging the points assigned to each cluster, then reassigns points to the nearest centroid. After many iterations on a large dataset, those centroid coordinates have been summed and divided enough times that small rounding errors could drift the result. Quiver works in squared distance internally to avoid unnecessary `sqrt` operations, uses stable summation for centroid updates, and compares positions with a tolerance during convergence checks. Two runs on the same data with the same seed return identical results. See <doc:KMeans-Clustering> for the clustering API and the convergence guarantees.

The `LinearRegression` model has a related concern. When feature columns are nearly redundant — say, height in inches and height in centimeters — the underlying matrix becomes ill-conditioned: technically invertible, but so close to non-invertible that floating-point noise gets amplified into large errors in the coefficient estimates. Quiver exposes `.conditionNumber` as the diagnostic. The <doc:Determinants-Primer> covers it in detail; a condition number above `10⁶` is the signal to stop and rethink. See <doc:Linear-Regression> for the model itself and the inference surface that surfaces the condition number alongside the fit. High-degree `polyfit` is the polynomial sibling of this problem — the Vandermonde columns it builds become ill-conditioned as the degree grows, for the same reason. See <doc:Polynomials>.

Two more defenses worth knowing by name. Multiplying many small probabilities together underflows to zero — Quiver works in log-space instead, where products become sums and tail densities stay representable. See `logPDF` on <doc:Working-With-Distributions> and the Naive-Bayes prediction path on <doc:Naive-Bayes> for the two sites that apply it. The mirror problem on the other end is `exp` overflowing on large inputs, which Quiver handles by subtracting the maximum value before exponentiating. See <doc:Activation-Functions> for the `softMax` site.

### Undefined results versus missing data

Quiver uses two different signals for two different situations. The Swift `nil` we already know means "there is no value here." `NaN` — short for *Not a Number* — means "there is a value here, but the math that produced it is undefined." They live at different layers entirely.

A `nil` value is Swift's way of saying the operation could not be performed. Calling `mean()` on an empty array returns `nil` because there is no data to average. The return type is `Double?` and we unwrap with `if let` or `guard let`, the same as any other optional in Swift.

A `NaN` is a value of type `Double` defined by the IEEE-754 floating-point standard, available in Swift as `Double.nan`. It is what mathematical operations return when the result is mathematically undefined — `0.0 / 0.0`, the square root of a negative number, or the correlation of a column with itself when that column has zero variance. The data was present; the math had no valid answer.

```swift
let mean = [Double]().mean()        // nil — no data to average
let undefined = 0.0 / 0.0           // NaN — division is undefined, but a Double came back

if let m = mean {
    print(m)                        // we unwrap nil with if let
}

if undefined.isNaN {
    print("undefined result")       // we test NaN with .isNaN
}
```

Three rules of `NaN` are worth carrying around. First, `NaN` propagates: any arithmetic that touches it produces `NaN`. Add five to `NaN`, multiply it by zero, take its square root — all `NaN`. This is the standard's way of refusing to invent a meaning for the calculation. Second, `NaN` is never equal to anything, including itself. The expression `Double.nan == Double.nan` is `false`. This means we never test for `NaN` with `==`; we use `.isNaN`. Third, Quiver leans on propagation deliberately rather than silently dropping `NaN` values from a calculation — a `NaN` reaching a result is the signal that something upstream needs cleaning, not a value to ignore.

The contract across Quiver is consistent. Functions that have no data to work with return `nil`. Functions that have data but cannot produce a finite mathematical result return `NaN`. The decision between them is "did the inputs exist" versus "did the math succeed."

> Experiment: **The Quiver Notebook** shows both signals on a fitness app. Try `heartRate.mean()` on an empty array — returns `nil`, no data to average. Then try `[72.0, 72.0, 72.0].correlation(with: stepCount)` on a flat heart rate — returns `NaN`, zero variance is a divide-by-zero. See <doc:Quiver-Notebook>.

### Order of operations matters

A surprising consequence of floating-point arithmetic is that it is not associative. The order in which we add a list of numbers can change the result:

```swift
let values = [1.0, 1e16, -1e16, 1.0]

// Left-to-right summation
values.reduce(0, +)
// 2.0 in exact arithmetic, but may compute as 0.0 due to cancellation
```

The notation `1e16` is Swift's shorthand for `1 × 10¹⁶`, ten quadrillion — the same scientific-notation pattern we met in the tolerance Tip earlier. When we add `1.0` to `1e16`, the `1.0` is so much smaller than `1e16` that it falls below the precision of the result. The `1.0` is effectively lost. Then we subtract `1e16`, leaving `0`, and add the final `1.0` to get `1.0`. The true answer in exact arithmetic is `2.0`.

This is why Quiver's aggregation functions — `mean`, `sum`, `standardDeviation` — are deliberate about the order in which they accumulate values. For most well-behaved data the order does not matter, but Quiver does not assume the input is well-behaved.

### When precision matters

Most numerical code does not need to worry about any of this. Adding a list of test scores, computing a mean response time, normalizing a feature column — these operate on data with limited dynamic range and produce results that are correct to many more digits than the application cares about. The floating-point approximation is invisible.

The cases where numerical error becomes visible share a few warning signs. The data spans many orders of magnitude — values from `1e-10` to `1e10` in the same array. The algorithm subtracts nearly equal numbers. The algorithm iterates many times, accumulating small updates. The matrix being inverted has a high condition number. Or the result is fed into a downstream system that compares it with strict equality.

When any of those conditions hold, the path forward is to check `.conditionNumber` where it applies, use Quiver's stable methods rather than rolling our own from textbook formulas, and never test floating-point results with `==`. Tolerance-based comparison — `abs(a - b) < tolerance` — is the floating-point equivalent of equality. The choice of tolerance depends on the application, but `1e-9` is a reasonable default for `Double` arithmetic on well-conditioned problems.

### Numerical literacy in practice

The goal of this primer is not to make every developer an expert in floating-point arithmetic. The goal is to recognize the warning signs early enough to ask the right question. When a result looks suspicious — a variance of zero on data that clearly varies, an inverted matrix that produces nonsense, two runs of the same algorithm that disagree by an amount that grows with the dataset size — the cause is often numerical, and the fix is often a different formulation of the same calculation.

Quiver's API is designed so that the default path is the numerically sound path. Calling `variance` is correct. Calling `standardDeviation` is correct. Inverting a well-conditioned matrix is correct. The places where we need to think — when to check `.conditionNumber`, when to use a tolerance instead of `==`, when to worry about summation order — are the places where the underlying mathematics demands thought, not the places where the API has cut corners.

> Experiment: **The Quiver Notebook** is the right place to see why this matters. Try `let x = [1_000_000.001, 1_000_000.002, 1_000_000.003]` and compute the variance two ways — the textbook formula `mean(x²) − mean(x)²` next to `x.variance()`. The textbook version returns a wrong answer, sometimes zero or even negative. Quiver's returns the small positive number we expect. The two should match; that they do not is the warning sign this primer is about. See <doc:Quiver-Notebook>.

