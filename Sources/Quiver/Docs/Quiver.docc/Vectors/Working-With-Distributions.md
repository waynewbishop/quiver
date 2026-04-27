# Working with Distributions

Evaluate probability densities, cumulative probabilities, and quantiles for named distributions.

## Overview

A **probability distribution** describes how likely each possible value of a random quantity is. Test scores cluster around an average and taper off in both directions. Sensor noise sits near zero with most samples small and a few large. Heights of adults fall in a bell-shaped band. The same mathematical machinery — a probability density function, a cumulative distribution function, and a quantile function — describes all three.

The `Distributions` namespace groups these functions by distribution name. Quiver ships the **normal distribution** at `Distributions.normal`, with `pdf`, `logPDF`, `cdf`, and `quantile` available as static methods. Each call passes the distribution parameters (`mean`, `std`) directly — there is no fitted-distribution object to construct, no shared state to mis-configure, and every call site is self-documenting.

```swift
import Quiver

// Density, cumulative probability, and quantile of a standard normal
Distributions.normal.pdf(x: 0, mean: 0, std: 1)         // ≈ 0.3989
Distributions.normal.cdf(x: 1.96, mean: 0, std: 1)      // ≈ 0.975
Distributions.normal.quantile(p: 0.975, mean: 0, std: 1) // ≈ 1.96
```

### Probability density and log-density

The **probability density function** (PDF) gives the relative likelihood of observing a specific value under the distribution. It is positive everywhere on the real line, peaks at the mean, and integrates to `1.0` across its full support. For a standard normal — `mean: 0, std: 1` — the density at the mean is approximately `0.3989`, which is `1 / √(2π)`:

```swift
import Quiver

// Peak density of a standard normal sits at the mean
let peak = Distributions.normal.pdf(x: 0, mean: 0, std: 1)  // ≈ 0.3989

// One standard deviation out — density drops by about 39%
let oneSigma = Distributions.normal.pdf(x: 1, mean: 0, std: 1)  // ≈ 0.2420
```

The **log-density** returns the natural log of the same quantity. Working in log-space is the standard tactic for numerical work that combines many density values together — products become sums, and densities far in the tail (which would round to zero in linear space) stay representable. `GaussianNaiveBayes` calls `Distributions.normal.logPDF` directly during prediction — the same implementation we expose publicly. Any classifier, kernel density estimator, or probabilistic model we write next can use the same well-tested function without reimplementing the math:

```swift
import Quiver

// Log-density 4σ from the mean — finite even though the linear density is tiny
let lp = Distributions.normal.logPDF(x: 4, mean: 0, std: 1)  // ≈ -8.919
```

### The cumulative distribution function

The **cumulative distribution function** (CDF) gives the probability that a normally distributed value falls at or below `x`. It rises monotonically from `0` at negative infinity to `1` at positive infinity, equals `0.5` at the mean, and at `x = 1.96` returns the canonical value used to construct 95% confidence intervals:

```swift
import Quiver

// Halfway up the distribution at the mean
Distributions.normal.cdf(x: 0, mean: 0, std: 1)  // = 0.5

// 1.96 standard deviations above the mean — the 97.5th percentile
Distributions.normal.cdf(x: 1.96, mean: 0, std: 1)  // ≈ 0.975
```

A worked example makes the value concrete. Consider a class where test scores are approximately normal with mean `75` and standard deviation `10`. What fraction of the class scored below `90`?

```swift
import Quiver

// Test scores: mean = 75, std = 10
let belowNinety = Distributions.normal.cdf(x: 90, mean: 75, std: 10)  // ≈ 0.9332
```

About 93% of the class scored below `90`. The CDF turns "where does this value sit in the distribution" into a concrete probability.

### Quantiles and critical values

The **quantile function** is the inverse of the CDF. It answers the question "what value of `x` puts probability `p` below it?" For a standard normal, the quantile at `p = 0.975` is approximately `1.96` — exactly the cutoff that puts 2.5% of the distribution in each tail and bounds a 95% confidence interval:

```swift
import Quiver

// 95% critical value — the canonical confidence-interval cutoff
Distributions.normal.quantile(p: 0.975, mean: 0, std: 1)  // ≈ 1.96

// And the median sits at the mean for a symmetric distribution
Distributions.normal.quantile(p: 0.5, mean: 0, std: 1)    // = 0
```

The same function works on non-standard normals by passing `mean` and `std`. Continuing the test-score example: what cutoff score does 90% of the class fall below?

```swift
import Quiver

// 90th percentile of test scores: mean = 75, std = 10
let cutoff = Distributions.normal.quantile(p: 0.90, mean: 75, std: 10)  // ≈ 87.82
```

Roughly 90% of the class scored below `87.8`. The quantile turns "what percentile do I want" into a concrete score on the original measurement scale.

### Why the optional return

Every function in `Distributions.normal` returns `Double?`. The optional makes out-of-domain input a `nil` rather than a runtime trap or a silently propagating `NaN`. Three conditions produce `nil`: a non-positive standard deviation (`std <= 0`), a probability outside `(0, 1)` for `quantile`, and any computation whose result is non-finite (`NaN` or `±infinity` from extreme underflow). This matches the pattern used by `mean`, `median`, and other Quiver statistics — invalid input is handled at the call site with `if let` or `guard let`, not buried inside the result.

```swift
import Quiver

// std must be positive — nil falls out cleanly
let bad = Distributions.normal.cdf(x: 1.0, mean: 0, std: -1)  // nil

// p must lie strictly between 0 and 1
let edge = Distributions.normal.quantile(p: 1.0, mean: 0, std: 1)  // nil
```

