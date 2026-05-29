# Regression Summary

Reading the standard errors, p-values, and confidence intervals returned by a fitted regression.

## Overview

After we have fitted a `LinearRegression` model on training data, `summary(features:targets:level:)` returns a `RegressionSummary` value carrying the inference table. The fit answers a prediction question — given features, the model returns the expected target. The summary answers an estimation question — given a fitted coefficient, the table tells us how confident to be that the underlying relationship is real. The vocabulary it uses — standard error, t-statistic, p-value, confidence interval — is the foundation of sample-to-population reasoning covered in <doc:Inferential-Statistics-Primer>. This article walks the printed summary field by field, naming what each column means, how the numbers relate, and which assumptions every claim rests on. For the fit-and-predict workflow itself, see <doc:Linear-Regression>.

### What the summary carries

The `summary` method consumes the same `features` and `targets` we trained the model on, plus an optional confidence level (default `0.95`), and returns one printable value:

```swift
let report = try model.summary(features: features, targets: targets)
print(report)
```

The printed output reads as one block. The top header carries the sample size `n`, the residual degrees of freedom `df`, the overall-fit metrics, and the residual standard error. The body table lists one row per coefficient — the intercept first when fitted, then the feature weights in input order:

```
Linear Regression Summary
=========================
n = 12, df = 8
R²    = 0.9894
Adj R² = 0.9854
Resid SE = 9584.4677

term       coef     std err       t   P>|t|     [95% lo      95% hi]
---------------------------------------------------------------------
x0    72972.68   47589.51   1.5334  0.1637  -36768.94   182714.30
x1       92.02      21.64   4.2520  0.0028      42.12      141.93
x2    17454.85    9363.45   1.8641  0.0993   -4137.30    39047.01
x3    -2719.02    1315.04  -2.0676  0.0725   -5751.51      313.47
```

### Reading the coefficient table

The **standard error** of a coefficient quantifies how much the estimate would vary if we drew a different training sample of the same size. The formula is `SE = √(σ²·diag((XᵀX)⁻¹))` where `σ² = RSS / (n − p)` is the estimated residual variance. A small SE means the estimate is precise; a large SE means a different sample could have moved the coefficient meaningfully. Standard errors live in the units of the coefficient itself — a slope of `92` dollars-per-square-foot with an SE of `22` is precise; the same slope with an SE of `60` is not.

The **t-statistic** is the ratio `coef / SE` — how many standard errors the estimate sits away from zero. A coefficient five standard errors from zero is hard to attribute to sampling noise; a coefficient one standard error from zero is not.

The **p-value** converts that t-statistic into a probability via the t-distribution with `df = n − p` degrees of freedom: `p = 2·(1 − F_t(|t|, df))`. The absolute value matters — the test is two-tailed because we care whether the coefficient is non-zero in either direction. A p-value of `0.0028` means a sample like ours would arise only `0.28%` of the time if the true coefficient were zero, which is taken as evidence that the underlying relationship is not zero.

The **confidence interval** answers the same question in coefficient units. The interval is `coef ± t_crit · SE`, where `t_crit = t.quantile(1 − α/2, df)` is the critical value at the chosen confidence level (default `α = 0.05`, so `1 − α/2 = 0.975`). An interval that does not contain zero corresponds to a p-value below the chosen `α`; an interval that straddles zero corresponds to a p-value above it. The two views read off the same standard error and the same critical value, just from different ends.

> Important: A p-value of `0.03` does not mean there is a `3%` chance the coefficient is zero, and a 95% confidence interval does not mean there is a 95% chance the true coefficient falls inside the interval. The p-value is the probability of observing data this extreme *if the coefficient were zero*; the confidence interval has 95% long-run coverage across hypothetical repeated sampling. The values describe a procedure's behaviour, not the probability of the underlying parameter.

### Confidence intervals and the level parameter

The `level` argument controls the confidence level of every interval in the table. The default is `0.95`. Asking for `0.99` widens every interval because we are demanding a procedure that captures the true coefficient in `99%` of repeated samples instead of `95%`:

```swift
let wider = try model.summary(features: features, targets: targets, level: 0.99)
```

Interval width scales as `1 / √n` through the `SE` formula. Doubling `n` shrinks every interval by about 30%; quadrupling shrinks them by 50%. The `1/√n` rule is why halving an interval requires four times the data, not twice. A small sample with wide intervals is the data telling us we have not seen enough to commit to the estimate.

### Adjusted R² and residual standard error

**Adjusted R²** modifies the plain R² to penalize parameter count: `1 − (1 − R²) · (n − 1) / (n − p)` where `p` is the number of fitted coefficients including the intercept. Adjusted R² is always less than R² when `p > 1` and `R² < 1`; the interesting case is when adjusted R² falls *as a feature is added*, which is the signal that the new feature does not justify its complexity. Plain R² mechanically rises whenever we add a feature, even pure noise; only adjusted R² can tell us when the addition is paying for itself.

**Residual standard error** is the square root of the residual variance: `√(RSS / df)`. It reads in the units of the target. A residual SE of `9584` on house prices means the typical prediction misses by about `$9,584` once the model's degrees of freedom are accounted for. RMSE on the same fit (available through `predictions.rootMeanSquaredError(actual:)`) divides by `n` instead of `df`, so the residual SE is slightly larger than RMSE — by exactly the factor `√(n / df)`.

### The assumptions every claim rests on

The standard errors, p-values, and confidence intervals above are computed assuming the data meets the Gauss-Markov conditions plus residual normality:

- **Linearity in parameters.** The model `y = θ₀ + θ₁x₁ + ...` is correctly specified. The features can be transformed (squared, logged, multiplied) before fitting, but the relationship between the features as supplied and the target must be linear.
- **Independence.** The training rows are independent observations. Time-series data, repeated measurements on the same subject, and clustered samples violate this assumption; the standard errors are too small when they do.
- **Constant residual variance** (homoscedasticity). The spread of the residuals does not depend on the predicted value. Heteroscedastic residuals — funnel shapes when plotted against the fitted values — make standard errors silently wrong.
- **No perfect collinearity.** The feature columns are linearly independent. Quiver enforces this hard: `fit` throws `MatrixError.singular` when `XᵀX` cannot be inverted. Near-singularity (very high condition number) does not throw, but it inflates standard errors without warning.
- **Normally distributed residuals.** Required for the t-distribution to be the right reference distribution for `β / SE`. The normality assumption is the most forgiving of the five: even modestly large samples satisfy it asymptotically through the central limit theorem applied to the coefficient estimator.

Visual diagnostics for these assumptions — residuals-vs-fitted plots, Q-Q plots, leverage statistics — are not part of the Quiver surface yet. They remain on the roadmap. Until they ship, the assumptions are stated for the reader to check against the underlying data by hand.

### Exporting the summary

`description` produces the formatted block shown above — that's what `print(report)` calls. Two additional formatters export the same data in formats that paste cleanly into a Pull Request comment or a spreadsheet:

```swift
let markdown = report.markdownTable()
let csv = report.csvRows()
```

The `markdownTable` formatter produces a GitHub-flavored markdown table with the same columns as the printed output. The `csvRows` formatter produces CSV with one row per coefficient plus a header row. Both formatters compose with Swift's file-writing APIs when the output is destined for disk.

### When summary throws

The same singular-matrix condition that makes `fit` throw also makes `summary` throw. Without a stable inverse of `XᵀX`, the variance-covariance matrix is unreliable, every standard error multiplies by `diag((XᵀX)⁻¹)` and would carry the unreliability silently. The throw is intentional: the caller learns immediately that inference is not available rather than reading a struct of meaningless numbers. The fix is the same as for `fit` — remove the redundant features that collapsed the design matrix and refit. See <doc:Linear-Regression> for the worked singular-matrix example.

### From the summary to deeper inference

The vocabulary here — standard error, t-statistic, p-value, confidence interval, degrees of freedom — is the foundation of sample-to-population reasoning. The <doc:Inferential-Statistics-Primer> covers that foundation in general terms, including the distinction between point estimates and interval estimates, the role of the sampling distribution, and the conditions under which a test's stated error rate is the actual long-run rate. For the model that produces the fit this article reads, see <doc:Linear-Regression>. For the iterative optimizer that solves regression problems without a closed form, see <doc:Gradient-Descent>.

> Experiment: **The Quiver Notebook** is the right place to watch standard errors widen as `n` shrinks. Fit a regression on the full dataset and print the summary; then refit on the first half of the rows and compare. Every standard error grows by roughly `√2`; every interval widens by the same factor; every p-value rises. The relationship between sample size and confidence is the most concrete in regression. See <doc:Quiver-Notebook>.

## Topics

### Type

- ``RegressionSummary``

### Coefficient table

- ``RegressionSummary/coefficients``
- ``RegressionSummary/standardErrors``
- ``RegressionSummary/tStatistics``
- ``RegressionSummary/pValues``
- ``RegressionSummary/confidenceIntervals``
- ``ConfidenceInterval``

### Overall fit

- ``RegressionSummary/rSquared``
- ``RegressionSummary/adjustedRSquared``
- ``RegressionSummary/residualStandardError``
- ``RegressionSummary/n``
- ``RegressionSummary/degreesOfFreedom``
- ``RegressionSummary/confidenceLevel``

### Formatters

- ``RegressionSummary/description``
- ``RegressionSummary/markdownTable()``
- ``RegressionSummary/csvRows()``

### Related

- <doc:Inferential-Statistics-Primer>
- <doc:Gradient-Descent>
