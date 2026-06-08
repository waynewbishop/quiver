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

The same care extends to iterative algorithms. The ``KMeans`` model recomputes cluster centroids by averaging the points assigned to each cluster, then reassigns points to the nearest centroid. After many iterations on a large dataset, those centroid coordinates have been summed and divided enough times that small rounding errors could drift the result. Quiver works in squared distance internally to avoid unnecessary `sqrt` operations, uses stable summation for centroid updates, and compares positions with a tolerance during convergence checks. Two runs on the same data with the same seed return identical results. See <doc:KMeans-Clustering> for the clustering API and the convergence guarantees.

### Verifying the fit

Sometimes the data we hand a model is shaped in a way the math cannot solve cleanly, no matter how careful Quiver is internally. The classic case is ``LinearRegression`` with two feature columns that say the same thing twice — height in inches and height in centimeters, for example. The model technically returns coefficients, but tiny changes to the input produce wildly different answers. The math is unstable, and Quiver cannot rescue it; the only honest move is to tell the caller. Quiver exposes `.conditionNumber` for exactly this — a single number that says how trustworthy the fit is. Above `10⁶`, the fit is not reliable and the right move is to drop one of the redundant columns and refit. See <doc:Linear-Regression> for the model, and <doc:Determinants-Primer> for the math behind the diagnostic.

The same shape of problem shows up in `polyfit` when we ask for a high-degree polynomial. Each new degree adds a column to an internal matrix that looks more and more like the columns next to it, and the fit becomes unstable for the same reason as a redundant feature. The fix is the same: drop the degree until the fit stabilizes. See <doc:Polynomials>, and see <doc:Rendering-Math-Primer> for how Quiver suppresses the leading machine-noise term in a rendered polynomial via the `relativeZeroTolerance` parameter.

### Defending against overflow and underflow

Floating-point numbers have a finite range as well as finite precision. Multiplying many small probabilities together underflows to zero — Quiver works in log-space instead, where products become sums and tail densities stay representable. See `logPDF` on <doc:Working-With-Distributions> and the Naive-Bayes prediction path on <doc:Naive-Bayes> for the two sites that apply it. The mirror problem on the other end is `exp` overflowing on large inputs, which Quiver handles by subtracting the maximum value before exponentiating. See <doc:Activation-Functions> for the `softMax` site.

### How Quiver signals an empty answer

A `nil` value is Swift's way of saying the operation could not be performed. Calling `mean()` on an empty array returns `nil` because there is no data to average. The return type is `Double?` and we unwrap with `if let` or `guard let`, the same as any other optional in Swift.

A `NaN` is a value of type `Double` defined by the IEEE-754 floating-point standard, available in Swift as `Double.nan`. It is what mathematical operations return when the result is mathematically undefined — `0.0 / 0.0`, the square root of a negative number, or an entry in a correlation matrix where one of the columns has zero variance. The data was present; the math had no valid answer.

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

The contract across Quiver follows a single rule. When the return slot can carry `nil` — a single scalar like `mean()`, `standardDeviation()`, or `correlation(with:)` — undefined math returns `nil`. When the return slot is a fixed-shape numeric container that cannot hold an optional in each cell — a correlation matrix, a regression's residual vector — undefined entries come back as `NaN`. The user-facing question is always "was the answer well-defined?" The answer arrives as `nil` where the type permits and as `NaN` where it does not.

> Experiment: **The Quiver Notebook** shows both signals on a fitness app. Try `heartRate.mean()` on an empty array — returns `nil`, no data to average. Then build a correlation matrix that includes a flat heart rate column — the matrix cells for that column come back as `NaN`, because zero variance makes the Pearson ratio undefined and a `[[Double]]` cannot carry an optional in each cell. See <doc:Quiver-Notebook>.

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

### Numerical literacy in practice

The goal of this primer is not to make every developer an expert in floating-point arithmetic. The goal is to recognize the warning signs early enough to ask the right question. When a result looks suspicious — a variance of zero on data that clearly varies, an inverted matrix that produces nonsense, two runs of the same algorithm that disagree by an amount that grows with the dataset size — the cause is often numerical, and the fix is often a different formulation of the same calculation.

Quiver's API is designed so that the default path is the numerically sound path. Calling `variance` is correct. Calling `standardDeviation` is correct. Inverting a well-conditioned matrix is correct. The places where we need to think — when to check `.conditionNumber`, when to use a tolerance instead of `==`, when to worry about summation order — are the places where the underlying mathematics demands thought, not the places where the API has cut corners.

> Experiment: **The Quiver Notebook** is the right place to see why this matters. Try `let x = [1_000_000.001, 1_000_000.002, 1_000_000.003]` and compute the variance two ways — the textbook formula `mean(x²) − mean(x)²` next to `x.variance()`. The textbook version returns a wrong answer, sometimes zero or even negative. Quiver's returns the small positive number we expect. The two should match; that they do not is the warning sign this primer is about. See <doc:Quiver-Notebook>.

