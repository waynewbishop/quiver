# The Central Limit Theorem

Building the sampling distribution of the mean.

## Overview

The Central Limit Theorem is the bridge between describing a dataset and reasoning about the population behind it. It explains why we can take a single sample, compute one mean, and still say something honest about a population we never fully observed.

This guide leads with the object the theorem is about: the sampling distribution of the mean. Once that object is in view, the standard error stops being a formula to memorize and becomes a quantity we can derive, and the distinction between standard deviation and standard error becomes a distinction we can state precisely.

### The sampling distribution of the mean

Imagine drawing the same-sized sample from a population over and over again. Each draw produces a slightly different sample mean. A different week of users gives a slightly different mean session time; a different watch session gives a slightly different step rhythm. The collection of all those possible sample means has its own distribution, the **sampling distribution of the mean**, and it describes how much the sample mean wobbles from one draw to the next.

This is the change in perspective the rest of the guide depends on. With a single sample, the mean is a number; we compute it once and it sits there. Across all the samples we could have drawn, the mean is a random variable with a distribution of its own. The sampling distribution of the mean is that distribution, and the Central Limit Theorem is a statement about its shape.

### The theorem

The **Central Limit Theorem** says that when we average many independent, identically distributed observations from a population with finite variance, the distribution of the sample mean approaches a normal, bell-shaped distribution regardless of the population's shape. Skewed populations, bimodal populations, populations with strange tails: once we average enough of them, the sample mean is approximately normal.

The practical payoff is that we do not need to know the shape of the population. We need a sample large enough for the theorem to apply, and from there the math we use on the sample mean is allowed to assume the bell-shaped behavior of a normal distribution. The next section makes the claim visible by building a population that is plainly not normal and watching its sample means form a bell anyway.

### Seeing it work

The theorem is easier to trust once we have seen it work. The snippet below builds a heavily skewed population from an exponential distribution, draws a thousand samples of size 50, and records the mean of each. The exponential distribution models time between events: minutes until the next customer arrives, seconds until the next request hits a server, days until a hard drive fails. Its `rate` parameter is the average number of events per unit time, so `rate = 0.5` means one event every two minutes on average, which is why the population mean equals `1 / rate = 2.0`.

The population is plainly not bell-shaped, since most values cluster near zero with a long right tail, but the distribution of sample means is:

```swift
import Quiver

// Build a skewed population with rate = 0.5, so the population mean is 1 / 0.5 = 2.0.
// Exponential is asymmetric: most values are small, with a long right tail.
// Seed the draw so the whole demonstration reproduces exactly.
var rng = SeededRandomNumberGenerator(seed: 7)
let population = [Double].randomExponential(10_000, rate: 0.5, using: &rng)

// Draw 1,000 samples of size 50 and record the mean of each.
let sampleMeans = population.samplingDistributionOfMean(
    sampleSize: 50,
    iterations: 1000,
    seed: 42
)

// The sampling distribution centers on the population mean, with its own spread.
sampleMeans.mean()              // ≈ 2.01  — recovers the population mean
sampleMeans.standardDeviation() // ≈ 0.29  — the standard error of the mean

// Confirm the bell shape: observed fractions match the Gaussian targets.
if let check = sampleMeans.empiricalRule() {
    print(check)
    // Empirical rule check (n = 1000)
    //               actual    expected    diff
    //   within 1σ:  0.678     0.683       -0.005
    //   within 2σ:  0.961     0.955       +0.006
    //   within 3σ:  0.995     0.997       -0.002
}
```

The mean of the sample means lands very close to the population mean of `2.0`, recovering it from a thousand draws without ever measuring the whole population. The `empiricalRule()` check is the bell test in code. It reports the fraction of sample means falling within one, two, and three standard deviations of their own mean, and those fractions sitting within a few thousandths of the Gaussian targets of `0.683`, `0.955`, and `0.997` is exactly what "approximately normal" means in practice.

<!-- DIAGRAM (fast-follow): skewed exponential population collapsing into a bell-shaped sampling distribution of the mean. Filename TBD by the visual team per the visual design guide. -->

### Why the spread shrinks with sample size

The bell shape is the theorem. The width of that bell is plain algebra, and it is worth doing the algebra first so the standard error formula arrives as a consequence rather than a rule. Variance adds across independent observations. The sample mean is the sum of `n` independent draws divided by `n`, so its variance is the population variance `σ²` divided by `n`: Var(X̄) = σ² / n. Taking the square root to return to the original units gives SD(X̄) = σ / √n. This holds for any population with finite variance, at any sample size; it does not wait for the bell curve to appear.

Put numbers to it with the exponential population above. The exponential standard deviation equals its mean, so the population standard deviation is `2.0`. The sample size is 50, and `2.0 / √50 ≈ 0.283`, which is the spread the simulation reported as roughly `0.29`. One slogan keeps the two ideas apart: the √n is algebra, the bell curve is the theorem.

> Note: The arithmetic above uses the population standard deviation, the `ddof: 0` form that divides by `n`. Quiver's `standardDeviation()` defaults to the sample form, `ddof: 1`, which divides by `n − 1`. The difference between the two shrinks as the sample grows and is negligible at the sample sizes where the Central Limit Theorem applies; pass `ddof: 0` explicitly when the population standard deviation is the quantity wanted.

### Two kinds of spread

The derivation names a second quantity. The spread of the sampling distribution of the mean has its own name, the **standard error** of the mean, and it is what `SD(X̄) = σ / √n` computes. The standard error is itself a standard deviation; it is the standard deviation of the sample mean rather than of the data, and the √n is the link between the two: SE = SD / √n.

It is tempting to call them the same quantity scaled, but that hides the distinction worth keeping. They measure different things. The standard deviation answers "how spread out are my data?"; the standard error answers "how spread out would my answer be?" The first is a property of the population and does not shrink with more data; the second is a property of the estimate and does shrink, because dividing by √n is what shrinks it. A small standard error means the sample mean barely moves from one hypothetical sample to the next; a large standard error means a single value should not be trusted on its own.

### Three instances of one idea

The mean is not the only statistic with a sampling distribution. Quiver ships `samplingDistributionOfMean`, `samplingDistributionOfMedian`, and `samplingDistributionOfStandardDeviation` as three instances of one idea: draw many samples, compute the statistic on each, and study how the statistic varies. Each returns a plain `[Double]`, so every Quiver statistic works on the result directly.

Comparing the mean's spread against the median's makes statistical efficiency concrete. On a normal population, the sampling distribution of the median is wider than that of the mean at the same sample size; the median's variance is larger by a factor of about π/2 ≈ 1.57, which is a spread factor of √(π/2) ≈ 1.25. The mean is the more efficient summary on this kind of data, meaning it extracts more precision from the same number of observations. The result is specific to a roughly normal population; on heavy-tailed data the comparison can reverse, which is why the demonstration below draws from `randomNormal`:

```swift
import Quiver

// A normal population: the setting where the mean's efficiency edge holds.
let population = [Double].randomNormal(10_000, mean: 50, standardDeviation: 8)

// Same population, same sample size, two different statistics.
let meanSpread = population
    .samplingDistributionOfMean(sampleSize: 40, iterations: 1000, seed: 1)
    .standardDeviation()    // ≈ 1.26 — the standard error of the mean

let medianSpread = population
    .samplingDistributionOfMedian(sampleSize: 40, iterations: 1000, seed: 1)
    .standardDeviation()    // ≈ 1.58 — wider, by roughly the √(π/2) factor
```

### From the theorem to inference

The standard error this guide derives is the input to every interval and test that follows. The <doc:Inferential-Statistics-Primer> puts it to work: it shows `standardError()` in a confidence interval, switches to the t-distribution when the sample is too small for the theorem to dominate, and uses resampling to build a sampling distribution directly from the data without assuming a shape. For the normal distribution that the theorem promises, with worked `pdf`, `cdf`, and `quantile` examples, see <doc:Working-With-Distributions>.

> Experiment: **The Quiver Notebook** is the right place to watch the standard error obey √n. Re-run the exponential demonstration with `sampleSize` set to `25`, `50`, `100`, and `200`, and read `standardDeviation()` off each `sampleMeans` array. Confirm that quadrupling the sample size from 50 to 200 halves the spread, since √4 = 2. The diminishing return turns from a memorized caveat into a measured curve. See <doc:Quiver-Notebook>.
