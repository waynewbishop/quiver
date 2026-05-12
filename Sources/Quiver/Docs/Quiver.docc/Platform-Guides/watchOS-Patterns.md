# watchOS Patterns

Turning hours of sensor data into a single score the user understands.

## Overview

Quiver gives us the building blocks to compose a single score from a session's worth of readings. A running app shows a workout score at the end of a run. A sleep app shows a sleep score in the morning. A focus app shows a focus score when a writing session ends. Each situation has the same shape — thousands of small readings on one side, one number the user remembers on the other, and a score that knows about variation and accumulation, not just averages.

### Summing effort and duration into a score

The simplest training-load score is a sum: for each sample, the effort label times the sample interval. A 30-minute steady run at "Moderate" effort accumulates differently from a 30-minute run at "Hard" effort because the effort multiplier differs even though the duration is the same.

```swift
import Quiver

// Effort labels (0=easy, 1=moderate, 2=tempo, 3=hard) sampled every 1.5 seconds
var efforts: [Double] = []
var deltaTimes: [Double] = []

func record(effort: Int, deltaTime: Double) {
    efforts.append(Double(effort))
    deltaTimes.append(deltaTime)
}

// At the end of the workout, sum the weighted contribution of every sample
func rawScore() -> Double {
    let weighted = zip(efforts, deltaTimes).map { $0 * $1 }
    return weighted.sum()
}
```

The `sum()` call is the foundation of the whole score. Everything else — variance multipliers, duration scaling, the cumulative load curve — is a transformation applied to the same sample stream that already contributes to this sum.

### Applying a variance multiplier

A steady run at "Tempo" effort and an interval workout that alternates "Easy" and "Hard" both produce a similar average effort, but they are not the same workout. The interval workout is harder. The variance multiplier captures that difference by adding a bonus proportional to how much the effort signal varies across the session.

```swift
import Quiver

// Variance of effort labels — higher when the workout includes intervals
let effortVariance = efforts.variance() ?? 0

// Bonus scales linearly with variance up to a 35% maximum
let varianceMultiplier = 1.0 + (0.35 * effortVariance / 2.25)
let adjustedScore = rawScore() * varianceMultiplier
```

The `variance()` call returns the sample variance of the effort labels using the default `ddof: 1`. The constant `2.25` is the variance of the most aggressive interval pattern the algorithm anticipates (alternating between 0 and 3), so the bonus saturates at 35% for genuinely interval-shaped workouts and contributes less for steady efforts. A steady run at constant "Tempo" produces an effort variance near zero and multiplier near 1.0 — the variance bonus is a no-op for steady efforts.

For richer variance accounting that ignores the initial warm-up — where effort is rising regardless of workout structure — the same `variance()` call applies to a slice:

```swift
// Exclude the first three minutes (≈120 samples at 1.5s cadence)
let warmupSamples = 120
let postWarmup = Array(efforts.dropFirst(warmupSamples))
let postWarmupVariance = postWarmup.variance() ?? 0
```

### Rendering a cumulative load curve

Workouts produce a load *over time*, not a single number at the end. The cumulative load curve answers "how much load have I accumulated so far?" at every point in the session, which is exactly what a watch face complication or a post-workout chart needs.

```swift
import Quiver

// Per-sample weighted contributions, accumulated into a never-decreasing curve
let weighted = zip(efforts, deltaTimes).map { $0 * $1 }
let loadCurve = weighted.cumulativeSum()
```

The `cumulativeSum()` call returns an array of the same length as the input, where each element is the sum of all preceding values plus the current one. The result is monotonically non-decreasing — every entry is at least as large as the one before it — because every weighted contribution is non-negative. That property is load-bearing for visualization: a chart of the load curve can never confuse the user by going backwards.

For a downsampled curve suitable for a watch-face complication, the same pattern composes with `downsample(factor:using:)` from the iOS-side data-visualization surface:

```swift
// Reduce 60 minutes of 1.5s samples (~2400 points) into 12 five-minute buckets
let perFiveMinute = loadCurve.downsample(factor: 200, using: .mean)
```

The downsampled curve preserves the cumulative shape — each bucket holds the average load at that point — and fits in the pixel budget a complication has available.

### Composing the pattern into a stateful aggregator

The three building blocks compose naturally into a stateful aggregator that records samples through the workout and produces a result at the end. The aggregator owns the buffers, exposes `record` and `pause`/`resume` methods to the workout UI, and computes the score and curve in a single `finalize()` call.

```swift
import Quiver

struct TESResult {
    let adjustedScore: Double
    let varianceMultiplier: Double
    let durationMinutes: Double
    let loadCurve: [Double]
}

final class TESAggregator {
    private var efforts: [Double] = []
    private var deltaTimes: [Double] = []
    private var isPaused = false

    func record(effort: Int, deltaTime: Double) {
        guard !isPaused else { return }
        efforts.append(Double(effort))
        deltaTimes.append(deltaTime)
    }

    func pause()  { isPaused = true }
    func resume() { isPaused = false }

    func finalize() -> TESResult? {
        guard efforts.count >= 40 else { return nil }   // need enough samples

        let weighted = zip(efforts, deltaTimes).map { $0 * $1 }
        let raw = weighted.sum()
        let curve = weighted.cumulativeSum()

        let variance = efforts.variance() ?? 0
        let multiplier = 1.0 + (0.35 * variance / 2.25)
        let durationMinutes = deltaTimes.sum() / 60.0

        return TESResult(
            adjustedScore: raw * multiplier,
            varianceMultiplier: multiplier,
            durationMinutes: durationMinutes,
            loadCurve: curve
        )
    }
}
```

Five Quiver methods (`sum`, `cumulativeSum`, `variance`, plus two more `sum` calls for the duration computation) do all the math. Everything else is bookkeeping — `pause`/`resume`, the minimum-sample threshold, the result struct that holds the outputs the UI needs.

### What this pattern is and what it is not

This is a stateful aggregator that accumulates labeled samples across a workout and produces a single score with a load curve. It is **not** a calibrated training-load model like TSS (Training Stress Score), CTL (Chronic Training Load), or ATL (Acute Training Load) from sports-science literature. Those models use decades-validated exponentially weighted averages over multiple workouts and require power-meter or pace-zone calibration. The TES pattern shown here is honest about what it captures: a single-workout score driven by the effort labels a classifier produced, with multipliers for variance and duration. For athletes who need calibrated training-load tracking across weeks of training, that is a separate problem with separate tools.

> Tip: The minimum-sample threshold (`>= 40` in the example, about one minute at a 1.5s cadence) protects against scoring a workout that barely happened. A user who starts a workout, walks 30 feet, and stops should not get a score; they did not work out. Pick a threshold that matches the shortest meaningful workout for the app's audience and enforce it at `finalize()` time.

> Experiment: [quiver-demo-tes-algorithm](https://github.com/waynewbishop/quiver-demo-tes-algorithm) is a reference implementation of the True Effort Score with fourteen scenario tests validated against real workouts. Running the test suite and comparing a steady thirty-minute Tempo workout (score 100.0) against an interval session of the same average intensity shows how the variance multiplier rewards spikiness — the intervals score higher even when the means match.
