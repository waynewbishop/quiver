# Correlation

Measuring how columns move together with Pearson correlation.

## Overview

We often want to ask one question of a dataset before any modeling begins: do these two columns move together? When the price of one product rises, do sales of another rise with it? When training volume goes up, does recovery time go down? Pearson correlation answers that question for the linear part of the relationship, returning a single number between `-1` and `+1` that says how tightly two columns track each other along a straight line.

### The Pearson formula

The Pearson product-moment correlation between two columns `x` and `y` is the covariance of the pair divided by the product of their standard deviations: `r = cov(x, y) / (sd(x) · sd(y))`. Covariance measures how the two columns vary together — large when they rise and fall in step, near zero when they move independently. Dividing by the product of the standard deviations rescales that joint variation into a unitless number: the units of each input cancel in the ratio, leaving a pure value that compares cleanly across datasets with very different scales. Because both pieces respond the same way to scale and shift, multiplying a column by ten or adding a constant to it leaves the correlation unchanged.

### How Quiver exposes correlation

Quiver exposes correlation at three levels. For two arrays, `[Double].correlation(with:)` returns the single Pearson coefficient directly. For a bare matrix, `correlationMatrix()` on `[[Double]]` computes every pairwise correlation in a `k`-by-`k` table. For a labeled table, `Panel.correlationMatrix()` returns the same numbers with the column names alongside, so a chart or report can label its axes from the same tuple it draws the numbers from.

### Pairwise correlation between two columns

When we have two `[Double]` inputs and want a single number, `correlation(with:)` returns it directly:

```swift
let x = [1.0, 2.0, 3.0, 4.0, 5.0]
let y = [2.0, 4.0, 5.0, 4.0, 5.0]

let r = x.correlation(with: y)   // Optional(0.7746) — moderately strong positive linear trend
```

The number `0.7746` says that `x` and `y` move together in the same direction most of the time, but not perfectly — the fourth point of `y` dips when `x` keeps rising, and that small disagreement pulls the correlation off `1.0`. The return is `Double?` because either vector having zero variance makes the Pearson ratio undefined; the convention matches `mean()` on an empty array and `standardDeviation()` when `n < 2`. The pairwise call agrees with the matrix-level result for the same pair, so unwrapping `x.correlation(with: y)` and reading `[x, y].correlationMatrix()[0][1]` produce identical numbers.

### Correlations across every pair in a panel

A panel of three columns produces a 3-by-3 matrix. The diagonal is `1.0` by construction — a column always correlates perfectly with itself. The off-diagonal entries are the numbers we are actually reading:

```swift
import Quiver

let panel = Panel([
    ("hours",   [1.0, 2.0, 3.0, 4.0, 5.0]),
    ("score",   [60.0, 70.0, 75.0, 85.0, 95.0]),
    ("fatigue", [85.0, 75.0, 60.0, 45.0, 25.0])
])

let result = panel.correlationMatrix()

// result.columns: ["hours", "score", "fatigue"]
// result.matrix:
// [[ 1.0000,  0.9948, -0.9934],
//  [ 0.9948,  1.0000, -0.9922],
//  [-0.9934, -0.9922,  1.0000]]
```

The returned tuple keeps the labels aligned with the rows and columns of the matrix. To read the correlation between `hours` and `fatigue`, we index `matrix[0][2]` — the first column listed paired with the third — and we expect a strongly negative number because more hours of practice correspond to less fatigue at the end of the session.

```swift
let hoursFatigue = result.matrix[0][2]   // -0.9934
let scoreFatigue = result.matrix[1][2]   // -0.9922
let hoursScore   = result.matrix[0][1]   //  0.9948

print(result.columns)          // ["hours", "score", "fatigue"]
print(result.matrix[0][2])     // -0.9934 — hours vs fatigue
```

Pairing the columns with the matrix in one return value means a chart or report can label its axes from the same tuple it draws the numbers from. The order of `columns` is the order the panel was built in, so a downstream caller does not have to look up names separately.

### What the diagonal and symmetry guarantee

Two structural properties hold for every correlation matrix that Quiver returns, and they are useful as quick sanity checks when the rest of a pipeline is uncertain.

The diagonal is always `1.0`. A column correlates perfectly with itself because the numerator and the denominator of the Pearson ratio reduce to the same sum of squared deviations. Glancing at the diagonal verifies that the matrix was built over the columns we expected and that none of them has been silently replaced by a different column.

The matrix is symmetric: `matrix[i][j]` equals `matrix[j][i]`. Pearson correlation is symmetric in its two arguments, because the covariance and the two standard deviations all treat the pair the same way regardless of order. In practice this means we only need to read the upper or lower triangle of the matrix — the other half is the same numbers reflected across the diagonal.

### Constant columns and NaN

A column whose values are all equal has zero variance, which places a zero in the denominator of the Pearson ratio. Quiver returns `Double.nan` for every entry in that column's row and column rather than substituting a zero or crashing. `0.0` would carry the meaning "no linear relationship", and that is not the same statement as "the correlation is undefined." NaN preserves the distinction and forces the caller to handle it explicitly. See <doc:Numerical-Literacy> for the contract between `nil` and `NaN` across Quiver.

```swift
let r = result.matrix[i][j]
if r.isNaN {
    print("\(result.columns[j]) has zero variance — correlation undefined.")
} else {
    chart(value: r)
}
```

The same rule applies when an input contains `NaN` already. The value propagates through the mean, the deviations, and the final ratio rather than getting silently dropped — a `NaN` reaching a correlation cell is the signal that something upstream needs cleaning.

### The limits of a linear measure

A correlation near `+1` or `-1` says that two columns move together in a straight line. It does not say that one causes the other, and it does not capture relationships that bend. A perfect parabola `y = x²` over a symmetric range produces a Pearson correlation of zero even though the two columns are deterministically related — the linear component cancels out. For monotonic but nonlinear associations, a rank-based statistic such as Spearman is the appropriate tool. Spearman is out of scope for `1.2.0`.

Correlation is also not the cosine similarity used in <doc:Similarity-Operations>. Both produce a number between `-1` and `+1`, and the formulas look similar, but cosine compares vectors as directions from the origin while correlation compares columns after centering them on their means. The two measures answer different engineering questions and they are not interchangeable.

A shift test makes the difference visible. Adding a constant to one column leaves correlation unchanged but moves cosine, because cosine sees the shifted arrow as a different direction:

```swift
let x = [1.0, 2.0, 3.0]
let y = [2.0, 3.0, 5.0]

x.cosineOfAngle(with: y)  // 0.9972
x.correlation(with: y)    // Optional(0.9820)

let yShifted = y + 10.0
x.cosineOfAngle(with: yShifted)  // 0.9564
x.correlation(with: yShifted)    // Optional(0.9820)
```

Correlation centers each column on its mean before measuring, so the constant shift is removed before the comparison.

### Why sample and population produce the same correlation

Elsewhere in Quiver, the choice between dividing by `n` and dividing by `n − 1` matters. The `variance(ddof:)` and `standardDeviation(ddof:)` methods expose the parameter so the caller can pick the sample or the population estimator. Correlation is the one place where the choice does not affect the result.

Pearson correlation is a ratio. The covariance in the numerator carries one factor of `1 / (n − ddof)`. Each standard deviation in the denominator carries a `1 / (n − ddof)` under its square root, and the two roots multiply to put one full factor of `1 / (n − ddof)` back into the denominator. The same constant scales the top and the bottom, and it cancels. A correlation computed with `ddof = 0` equals the correlation computed with `ddof = 1` to floating-point precision. The `ddof` convention for the rest of the statistics surface is described on <doc:Statistics-Primer>.

### Complexity

A panel with `k` columns and `n` rows produces a `k`-by-`k` correlation matrix. Each cell is a pairwise correlation that walks both columns once, so the overall cost is `O(k² · n)`. The symmetric structure lets a future optimization compute only the upper triangle, but the public contract is the full matrix today.

For a panel with five columns and a thousand rows, this is twenty-five pairwise passes of a thousand elements each — fast enough to compute interactively at every cell of a dashboard. The cost grows quadratically in the number of columns, so a panel with two hundred columns and a million rows is where we would want to think about chunking the work.

## Topics

### Computing correlations

- ``Swift/Array/correlation(with:)``
- ``Panel/correlationMatrix()``
- ``Swift/Array/correlationMatrix()``
