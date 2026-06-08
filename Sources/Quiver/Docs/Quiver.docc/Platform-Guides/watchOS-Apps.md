# watchOS Guide

Build a personal baseline, measure today's session against it, and fit a model from the user's own telemetry on Apple Watch.

## Overview

On watchOS, statistics, linear algebra, and machine learning share a single purpose: building the user's **personal baseline**. A baseline is what the watch knows about its wearer: typical pace, resting heart rate, movement signature. Statistics describes the baseline. Linear algebra measures how today's session sits relative to it. Machine learning fits a model of it.

The same pattern holds across every athletic discipline: running, cycling, swimming, hiking, climbing, and strength training. We turn data from the wrist into a model of the wearer, compute it on the wearer's own watch, and keep it owned by the wearer, never leaving the device.

### Setup and lifecycle

A watchOS app reads samples from `HKWorkoutSession` and `HKAnchoredObjectQuery`, decodes them into `[Double]` at the boundary, and feeds them to Quiver. Anything that should outlive a single workout — a baseline, a fitted classifier, a stored set of feature vectors — encodes to JSON at session end and writes to `Documents/`. The next session decodes the same value at launch. Every model and baseline is `Codable` and `Sendable`, so it crosses task boundaries during a live session and persists across sessions without ceremony. See <doc:Model-Persistence> for the encode-and-decode shape every persisted value on the watch shares.

## Statistics on watchOS

A workout produces a stream of samples: heart rate every second, pace every stride, power every cadence cycle. Sixty heart-rate samples are not a workout summary. The job of descriptive statistics here is to compare a fresh window of samples against the wearer's own baseline, so the watch can say "this is hard for you" instead of "this is 168 BPM." Quiver computes the baseline once, stores the typed snapshot to disk, and reads it back at session start to score every incoming sample.

### The baseline and its summary

That comparison requires the baseline introduced above, a stored description of what is normal for this wearer, and two summary statistics describe its core. The **mean** is the arithmetic average of a buffer; it answers "what is typical right now." The **standard deviation** is the typical distance of a sample from the mean; it answers "how steady is the current effort." A standard deviation that doubles between mile one and mile five is a stability signal the app can act on, even without knowing what the absolute pace number means. See <doc:Statistics-Primer> for the vocabulary of central tendency and spread, and <doc:Frequency-Tables> for the categorical analog when the baseline counts events rather than averages values.

The lead example is running pace stability. A 60-second buffer of pace samples gives a standard deviation that measures how steady the runner is holding pace. The `empiricalRule()` call compares the current buffer's distribution against the wearer's personal baseline.

```swift
import Quiver

// Ten pace samples (m/s) from one window of a steady run.
let paceBuffer = [3.5, 3.3, 3.4, 3.6, 3.2, 3.5, 3.4, 3.3, 3.6, 3.2]

guard let mean = paceBuffer.mean(),
      let std = paceBuffer.standardDeviation(),
      let rule = paceBuffer.empiricalRule() else { return }

print(mean)                // 3.4 — typical pace for this window
print(std)                 // 0.149 — pace varied by about 0.15 m/s sample-to-sample
print(rule.within1Sigma)   // 0.60 — six of ten samples sat within one std of the mean
print(rule.expected1Sigma) // 0.6827 — what a normal distribution predicts
```

### Reading the gap as a signal

The `empiricalRule()` method returns the fraction of samples that fall within one, two, and three standard deviations of the mean, alongside the theoretical fractions a normal distribution predicts. A real workout produces real numbers; the theoretical numbers are what a textbook predicts. The interesting product signal is the gap between them. When the within-1σ fraction sits noticeably below 68.27%, the underlying distribution is not normal, and that gap itself is the signal: fatigue, sensor noise, or a session that does not belong in the baseline. See <doc:Inferential-Statistics-Primer> for the framework that lets the app act on a gap of that size with confidence.

> Note: When the watch reports a within-1σ fraction far below the theoretical 68.27%, the underlying distribution is not normal. That gap is the signal: fatigue, sensor noise, or a session that does not belong in the baseline.

### Generalizing across disciplines

The same shape generalizes across disciplines. A standard deviation over a rolling buffer of pace, power, stroke rate, or HR-recovery samples reads as a stability measure for the current effort, and a percentile against the wearer's stored baseline answers whether the current value sits in the wearer's normal range. The math does not change when the sport changes; only the column being summarized does.

## Linear algebra on watchOS

The watch sits on a wrist that is constantly moving, and a three-axis accelerometer records that movement as a stream of `[x, y, z]` triples. A triple by itself is just three numbers. What gives it meaning is geometry: each triple is a **vector**, a directed arrow in three-dimensional space, and the things the app cares about — orientation, symmetry, alignment — are answered by geometric operations on those arrows.

### Magnitude and direction

The `magnitude` of a vector is its length. The `normalized` form is the same arrow scaled to length 1, which strips away "how strong" and keeps "which direction." The `dot` product of two unit vectors is the cosine of the angle between them, which answers "how aligned are these two directions." With those three primitives we turn raw accelerometer data into the angle of a trail, the symmetry of a gait, or the coupling between cadence and power. See <doc:Linear-Algebra-Primer> for the geometric foundation and <doc:Vector-Operations> for the operations Quiver exposes on `[Double]`.

The lead example is hiking incline from gravity. At rest, the accelerometer reading points straight down. Gravity is the strongest constant signal on the wrist, and once we normalize it to a unit vector, the watch effectively becomes a pointer. The angle between that pointer and the vertical reference is the slope of the surface beneath the wearer.

```swift
import Quiver

// A 3-axis accelerometer reading taken on a hike.
let sample: [Double] = [1.0, 1.36, 9.66]

let gravity = sample.normalized
let vertical: [Double] = [0, 0, 1]
print(gravity.dot(vertical)) // ≈ 0.985 — cosine of the angle between the watch and vertical
```

The dot product of the normalized gravity vector and the vertical reference returned `0.985` — that is the cosine of the angle between the watch and vertical. A flat surface scores close to `1.0`; a vertical wall scores close to `0`.

### From cosine to degrees

Reading that cosine as a slope in degrees is the standard Foundation math step that follows. The cosine is what the math returns; degrees are what the wearer sees on screen.

```swift
import Foundation

let cosTheta = 0.985 // from the previous block
let inclineRadians = acos(cosTheta)
print(inclineRadians * 180 / .pi) // ≈ 9.9° — the slope of the trail
```

A few lines of vector math turn a raw sensor reading into the angle of the trail. The same primitives — `magnitude`, `dot`, `normalized`, `cosineOfAngle` — drive activity-specific features across disciplines, including gait symmetry, cadence-power coupling, and stroke geometry, all built from short fixed-dimension vectors over windowed sensor data. <doc:Vector-Projections> covers the related operation that decomposes one vector into a component along another, which is the math under stride-direction and impact-axis features.

### Physical signals from sensor windows

These primitives compose into a deeper signal-processing surface. `powerSpectralDensity` and `trapezoidalIntegral` detect rhythmic motion, cumulative effort, and frequency content from the same `[Double]` arrays the dot-product examples above use. See <doc:Physics-Primitives-Primer> for the domain-units framing — when an integral of acceleration over time has the units of velocity and when it does not.

> Tip: These primitives compose into a deeper signal-processing surface — `powerSpectralDensity` and `trapezoidalIntegral` — that detects rhythmic motion, cumulative effort, and frequency content from the same `[Double]` arrays. See <doc:Physics-Primitives-Primer>.

> Note: For HR-derived signal analysis in the respiratory band (0.15–0.40 Hz), the spectral resolution `Δf = sampleRate / paddedLength` must reach 0.0083 Hz before the band is legible. A window shorter than 60 seconds collapses the resolution and reads as noise.

## Machine learning on watchOS

The watch's job in a machine-learning workflow is narrower than iOS's. The data is small, the timer budget is short, and the model has to be personal: fit from one wearer's telemetry, owned by that wearer, never leaving the wrist. See <doc:Machine-Learning-Primer> for the features-labels-training-evaluation framing that the rest of this section assumes.

### Closed-form fits on small data

Two methods cover most of the surface. <doc:Linear-Regression> fits a slope of one sensor against another using the closed-form **normal equation**, a single matrix calculation that finds the best line in one pass without iteration. <doc:Naive-Bayes> fits a classifier from a short calibration session by computing the mean and variance of each feature per labeled class. Both methods finish quickly enough to run between samples and produce a fitted value the watch can persist alongside the baseline.

The lead example is a personal HR-vs-power regression for cycling. At the end of a ride, we fit a linear regression of HR against power on the workout's samples. The fitted slope is the wearer's HR-per-watt sensitivity, a baseline that varies hugely between individuals.

```swift
import Quiver

// Seven aligned samples from one ride — power (watts) as the feature, HR (bpm) as the target.
let power: [[Double]] = [[100], [125], [150], [175], [225], [250], [275]]
let hr: [Double] = [100, 110, 120, 130, 150, 160, 170]

let model = try LinearRegression.fit(features: power, targets: hr)

// Score 200 watts against this wearer's fitted slope.
print(model.predict(200.0)) // 140.0 — expected HR at 200W for this wearer
```

The fitted `model` is a value with a `predict` method. Passing in 200 watts returns the heart rate the wearer's slope expects at that effort — here, 140 bpm. A different wearer's data would fit a different slope and a different intercept, and the predicted HR at 200W would land somewhere else; the model is *personal* in exactly that sense.

### Persisting the model across sessions

A model fit at the end of one ride is only useful if the next ride can read it. The fitted value is `Codable`, which means one `JSONEncoder` call turns it into bytes the watch can write to `Documents/` and decode at the next launch. See <doc:Model-Persistence> for the encode-once, decode-on-launch shape every persisted model on the watch shares.

```swift
import Quiver
import Foundation

// Refit the same HR-vs-power model from the previous block so this block stands alone.
let power: [[Double]] = [[100], [125], [150], [175], [225], [250], [275]]
let hr: [Double] = [100, 110, 120, 130, 150, 160, 170]
let model = try LinearRegression.fit(features: power, targets: hr)

// Encode the fitted value and write it to the Documents directory.
let documentsURL = FileManager.default
    .urls(for: .documentDirectory, in: .userDomainMask)[0]

let encoded = try JSONEncoder().encode(model)
try encoded.write(to: documentsURL.appendingPathComponent("hr-power.json"))
```

The next ride decodes the stored model and checks whether today's HR-at-200W is in line with the wearer's historical baseline. A large residual against the stored slope is the signal: fatigue, dehydration, or a fitness shift.

### Activity classification from a calibration session

A second pattern covers activity classification. Features from a fixed-length sensor window — rate, magnitude, symmetry — aggregate into a feature vector, `GaussianNaiveBayes` fits from a short user-labeled calibration session, and the watch predicts the activity type during the workout. See <doc:Naive-Bayes> for the probabilistic framing of how the per-class mean and variance score a new feature vector. The fit-on-the-wrist shape captures a different personal artifact for each discipline: an HR-vs-power curve, a walk-vs-hike-vs-climb classifier, a stroke-type model. Each one persists to Documents as a `Codable` value the next session can decode unchanged.

> Note: Closed-form methods on small per-user data are what the watch is for. Iterative-convergence methods over bulk data — `KMeans` with high `maxIterations`, `KNearestNeighbors` over thousands of training points, `Matrix.invert` on large dimensions — are not. The 30-second watchOS timer budget and the watch FPU's precision profile can return non-converged results that look correct. Move iterative-bulk fitting to iOS or the server.

## Where to go from here

The three sections above each have a deeper layer of math underneath them, and that math is the next step for watchOS developers moving into numerical work. <doc:Statistics-Primer> builds the vocabulary of variance, distributions, and the empirical rule that the baseline section leans on. <doc:Linear-Algebra-Primer> extends vectors and dot products into the geometric operations that read sensor data as orientation and alignment. <doc:Physics-Primitives-Primer> covers the signal-processing surface — integrals, derivatives, and frequency content — that turns short sensor windows into physical quantities. <doc:Machine-Learning-Primer> closes the loop with features, labels, training, and the trade-offs that decide which closed-form method to reach for on the wrist.

> Experiment: **The Quiver Notebook** is the right place to feel how `empiricalRule()` reads against personal baselines. Load a vector of recent HR samples, compare the observed within-1σ fraction to the theoretical 68.27%, and watch the gap shift as the workout type changes. The same `Codable` model that fits in the Notebook decodes unchanged on the watch. See <doc:Quiver-Notebook>.
