# Random Sampling

Drawing reproducible random samples from any array, with or without replacement.

## Overview

We want to know the average commute in a city, but we cannot stop every commuter and ask. We reach the handful who answer a survey, and from those few responses we estimate a number that describes everyone. This is the core move of applied statistics: a **sample** stands in for a population we cannot measure in full. Quiver provides `sample(_:replace:seed:)` as an extension on `Array`, drawing a random subset of any array so the subset can be inspected, summarized, and reasoned about directly.

A sample drawn **without replacement** takes each element at most once, the way a survey reaches each person a single time. A sample drawn **with replacement** returns each draw before the next, so the same element can be selected more than once: the basis for the resampling techniques that follow later.

### Basic usage

Suppose we surveyed thirty commuters and recorded their travel times in minutes. In the field we would reach only a few of them; here we hold all thirty so we can check our work against the truth. Draw five responses without replacement and take their mean as a point estimate of the citywide average:

```swift
import Quiver

let commuteMinutes = [22.0, 35, 28, 41, 19, 33, 26, 47, 24, 38,
                      31, 29, 52, 23, 36, 27, 44, 21, 34, 30,
                      25, 39, 20, 32, 48, 28, 37, 23, 31, 42]

let sampled = commuteMinutes.sample(5, replace: false, seed: 7)
let estimate = sampled.mean()   // 29.0 — one estimate from five responses
```

The method returns a plain `[Element]` (`[22.0, 23.0, 32.0, 27.0, 41.0]` for the call above), so every Quiver operation applies to the result without conversion. A `count` of `0` returns an empty array, and an empty source array returns an empty sample.

### Why sample when we have the data

The full array of thirty values is sitting right there, so a fair question arises: why not call `mean()` on all thirty and skip the sampling entirely? In this article the population exists for one reason: it is the answer key. We can only judge whether an estimate from five responses is any good because we happen to know the true average is about `32.17`, and the estimate of `29.0` falls a few minutes short of it.

In real work that answer key never exists. A survey reaches the people who reply and no others. Destructive testing consumes every unit it measures. Polling, forecasting, and controlled experiments all produce a sample and nothing more. The sample is the only data we will ever hold, and the rest of this article is about how to reason carefully from it: how far one estimate might land from the truth, and how to make it land closer.

### Sampling with replacement

The opening survey reached each commuter once, so it drew without replacement. Many statistical methods need the opposite: repeated independent draws from the same population, where a value may appear more than once. Passing `replace: true` allows repetition, which also means the sample may be larger than the source array:

```swift
import Quiver

let commuteMinutes = [22.0, 35, 28, 41, 19, 33, 26, 47, 24, 38,
                      31, 29, 52, 23, 36, 27, 44, 21, 34, 30,
                      25, 39, 20, 32, 48, 28, 37, 23, 31, 42]

let withReplacement = commuteMinutes.sample(8, replace: true, seed: 7)
// [22.0, 23.0, 20.0, 36.0, 23.0, 39.0, 24.0, 30.0] — the value 23.0 appears twice
```

> Note: Sampling without replacement is drawing names from a hat and keeping each one — every element is selected at most once, so the sample cannot be larger than the array. Sampling with replacement returns each draw to the hat before the next, so the same element can appear again and a sample larger than the source is allowed.

### Reproducible samples with seeds

The `seed` parameter fixes the draw. The same array with the same seed always produces the same sample, which is what makes an experiment re-runnable: running the analysis twice yields identical results rather than a new random subset each time:

```swift
import Quiver

let commuteMinutes = [22.0, 35, 28, 41, 19, 33, 26, 47, 24, 38,
                      31, 29, 52, 23, 36, 27, 44, 21, 34, 30,
                      25, 39, 20, 32, 48, 28, 37, 23, 31, 42]

let first = commuteMinutes.sample(5, replace: false, seed: 7)
let second = commuteMinutes.sample(5, replace: false, seed: 7)

// first == second  ✓ — same seed, same draw
```

Changing the seed produces a different subset, which lets us draw many independent samples and study how much the estimate moves from one to the next.

> Important: A sample drawn without replacement cannot contain more elements than the array holds. Requesting more triggers a runtime precondition failure rather than a truncated result, so check the count before drawing when the sample size comes from outside the code. Sampling with replacement has no such ceiling.

> Note: A given seed reproduces the same draw across runs and across platforms within Quiver. The sequence belongs to Quiver's own generator — treat the seed as a label for "the same Quiver result," not as a portable key that any other tool would reproduce.

### Using a custom generator

A second overload accepts a `RandomNumberGenerator` directly instead of a seed. This mirrors the standard library's `shuffled(using:)` and lets a single generator drive several draws in sequence, so each draw advances the same stream:

```swift
import Quiver

let commuteMinutes = [22.0, 35, 28, 41, 19, 33, 26, 47, 24, 38,
                      31, 29, 52, 23, 36, 27, 44, 21, 34, 30,
                      25, 39, 20, 32, 48, 28, 37, 23, 31, 42]

var generator = SeededRandomNumberGenerator(seed: 7)
let drawA = commuteMinutes.sample(5, replace: true, using: &generator)
let drawB = commuteMinutes.sample(5, replace: true, using: &generator)
// drawA and drawB differ — the generator advanced between them
```

> Tip: Reach for the `using:` overload when one draw must share a generator with the rest of a pipeline. The `seed:` overload is the simpler choice when a single reproducible draw is all that is needed.

### Works with any element type

Because drawing a sample is pure index selection, `sample` has no constraint on the array's elements. The method works on `[Double]`, `[String]`, `[[Double]]`, or any other Swift type:

```swift
import Quiver

let districts = ["north", "south", "east", "west", "central"]
let surveyed = districts.sample(2, replace: false, seed: 3)
// ["north", "south"]
```

### The building block for resampling

A single estimate could be lucky or unlucky, so the natural next step is to draw many samples and watch how the estimate behaves. Repeating a draw thousands of times and recording the mean of each builds a **sampling distribution**, and the spread of that distribution measures how much to trust any one estimate. The mean of two thousand sample means lands almost exactly on the true average, regardless of sample size:

```swift
import Quiver

let commuteMinutes = [22.0, 35, 28, 41, 19, 33, 26, 47, 24, 38,
                      31, 29, 52, 23, 36, 27, 44, 21, 34, 30,
                      25, 39, 20, 32, 48, 28, 37, 23, 31, 42]

let n5 = commuteMinutes.samplingDistributionOfMean(sampleSize: 5, iterations: 2000, seed: 42)
let n15 = commuteMinutes.samplingDistributionOfMean(sampleSize: 15, iterations: 2000, seed: 42)

n5.mean()   // ≈ 32.31 — close to the true mean of 32.17
n15.mean()  // ≈ 32.23 — also close, at a larger sample size
```

What changes with sample size is not the center but the spread. The standard deviation of the sample means, the standard error, nearly halves as the sample grows from five to fifteen, and it tracks the theoretical `stdev / √n` closely:

```swift
n5.standardDeviation()   // ≈ 3.96 — close to 8.83 / √5  ≈ 3.95
n15.standardDeviation()  // ≈ 2.31 — close to 8.83 / √15 ≈ 2.28
```

A larger sample buys a tighter estimate, and sample size is the lever we turn to control precision. The draw shown here is also the operation underneath several other methods: a <doc:Train-Test-Split> is a without-replacement draw specialized for splitting labeled data, and the bootstrap and sampling-distribution methods in <doc:Inferential-Statistics-Primer> and <doc:Central-Limit-Theorem> draw with replacement on every iteration.

> Experiment: **The Quiver Notebook** is the right place to feel the difference replacement makes. Draw a sample of size five without replacement from a ten-element array, then draw a sample of size twenty with replacement from the same array. The first can never repeat a value; the second will repeat several. Now draw twice without replacement using the same seed and watch the two draws match element for element. See <doc:Quiver-Notebook>.

## Topics

### Sampling
- ``Swift/Array/sample(_:replace:seed:)``
- ``Swift/Array/sample(_:replace:using:)``

### Builds on this
- <doc:Train-Test-Split>
- <doc:Inferential-Statistics-Primer>
- <doc:Central-Limit-Theorem>

### Related
- <doc:Random-Number-Generation>
