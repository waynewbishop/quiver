# watchOS Guide

Analyzing sessions, measuring them geometrically, and fitting models from Apple Watch telemetry.

## Overview

On watchOS, we use [statistics](<doc:Statistics-Primer>), [linear algebra](<doc:Linear-Algebra-Primer>), and [machine learning](<doc:Machine-Learning-Primer>) to build the user’s **personal baseline**. Statistics describes the baseline, linear algebra measures how today's session sits relative to it and machine learning fits a model. These actions run entirely on-device with Quiver, ensuring data privacy and high performance.

> Note: This guide builds on the concepts found in <doc:Statistics-Primer>.

### Setup and lifecycle

An app reads samples from `HKWorkoutSession` and `HKAnchoredObjectQuery`, decodes them into `[Double]`, and feeds them to Quiver. Anything that should outlive a single workout—a baseline, fitted classifier, or set of feature vectors—encodes to JSON and writes to `Documents/`. The next session decodes the value at launch. Models are `Codable` and `Sendable`, allowing them to cross task boundaries and persist without ceremony. See <doc:Model-Persistence> for the shared persistence pattern.


## Statistics on watchOS

Workouts stream samples: heart rate, pace, power. Descriptive statistics compare a fresh window of samples against the wearer's personal baseline to detect deviations. Quiver computes the baseline, stores the snapshot, and reads it back to score incoming samples.

### The baseline and its summary

Two statistics describe the core baseline. The **mean** (typical value) and **standard deviation** (typical variance). A standard deviation that doubles between mile one and mile five signals a stability shift. See <doc:Statistics-Primer> for center and spread, and <doc:Frequency-Tables> for categorical history.

The following example measures pace stability:

```swift
import Quiver

// Ten pace samples (m/s) from one window of a steady run.
let paceBuffer = [3.5, 3.3, 3.4, 3.6, 3.2, 3.5, 3.4, 3.3, 3.6, 3.2]

guard let mean = paceBuffer.mean(),
      let std = paceBuffer.standardDeviation(),
      let rule = paceBuffer.empiricalRule() else { return }

print(mean)                // 3.4 — typical pace
print(std)                 // 0.149 — pace variance
print(rule.within1Sigma)   // 0.60 — actual 1σ fraction
print(rule.expected1Sigma) // 0.6827 — expected 1σ fraction
```

### Reading the gap as a signal

The `empiricalRule()` method compares observed sample fractions within 1σ, 2σ, and 3σ against expected normal distributions. A significant gap—where observed fractions fall below expected ones—indicates the window is not in a steady state, flagging terrain changes or surges rather than diagnosing their cause. See <doc:Inferential-Statistics-Primer> for the framework to act on these gaps confidently.

> Note: A within-1σ fraction far below 68.27% means the buffered effort is not steady or normally distributed. That gap is a flag to investigate—not a diagnosis.


### Generalizing across disciplines

The same shape generalizes across disciplines. A standard deviation over a rolling buffer of pace, power, stroke rate, or HR-recovery samples reads as a stability measure for the current signal, and a percentile against the wearer's stored baseline answers whether the current value sits in the wearer's normal range. The math does not change when the sport changes; only the column being summarized does.

## Linear algebra on watchOS

An accelerometer records movement as a stream of `[x, y, z]` vectors. Geometry—specifically orientation, symmetry, and alignment—is answered by geometric operations on these vectors.

### Magnitude and direction

The `magnitude` of a vector is its length. `normalized` scales an arrow to length 1, focusing on direction. The `dot` product of two unit vectors yields the cosine of the angle between them, indicating alignment. With these primitives, we turn raw accelerometer data into trail slope, gait symmetry, or the coupling between cadence and power. See <doc:Linear-Algebra-Primer> for the geometric foundation and <doc:Vector-Operations> for the API.

To hike incline from gravity, normalize the gravity reading to a unit vector. The dot product of this vector and the vertical reference provides the slope:

```swift
import Quiver

// A 3-axis accelerometer reading taken with the wrist momentarily still.
let sample: [Double] = [1.0, 1.36, 9.66]

let gravity = sample.normalized
let vertical: [Double] = [0, 0, 1]
print(gravity.dot(vertical)) // ≈ 0.985 — cosine of the angle
```

A flat surface scores close to `1.0`; a vertical wall scores close to `0`. 

> Note: For noisy mid-stride readings, average over a short still window or use Core Motion’s `gravity` vector before trusting the angle.

### From cosine to degrees

Convert the cosine to degrees to match standard reporting:

```swift
import Foundation

let cosTheta = 0.985
let inclineRadians = acos(cosTheta)
print(inclineRadians * 180 / .pi) // ≈ 9.9° — trail slope
```

These primitives—`magnitude`, `dot`, `normalized`, `cosineOfAngle`—drive features across disciplines, including gait symmetry and stroke geometry, all built from short fixed-dimension vectors over windowed sensor data. <doc:Vector-Projections> covers the related operation that decomposes one vector into a component along another, which is the math under stride-direction and impact-axis features.

### Physical signals from sensor windows

These primitives compose into a broader signal-processing surface. `powerSpectralDensity` and `trapezoidalIntegral` extract rhythmic motion, cumulative effort, and frequency content from sensor data. See <doc:Physics-Primitives-Primer> for the physical interpretation of these signals.

> Tip: These primitives compose into `powerSpectralDensity` and `trapezoidalIntegral` to extract motion and effort metrics. See <doc:Physics-Primitives-Primer>.

> Note: An HR-derived signal in the respiratory band (0.15–0.40 Hz) needs spectral resolution `Δf = sampleRate / paddedLength` fine enough to separate structure. A 120-second window is typically required to avoid blurring the signal.


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

With a single feature there is no overlap between predictors to destabilize the fit, so the closed-form least-squares line is exact and stable. The moment several correlated signals enter the same fit — power alongside pace and grade, which carry much of the same information — the plain fit turns unstable and the regression needs a regularized form (`Ridge`) instead. That multi-signal case is worked end to end in <doc:Building-An-Effort-Model>, the effort-model demonstration this guide accompanies.

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

The next ride decodes the stored model and checks whether today's HR-at-200W lines up with the wearer's historical slope. A large residual flags that today's HR-to-power relationship has shifted from baseline — heat, altitude, cardiac drift, caffeine, or a genuine fitness change can all move it. The residual surfaces the divergence; interpreting it is the wearer's call.

### Activity classification from a calibration session

A second pattern covers activity classification. Features from a fixed-length sensor window — rate, magnitude, symmetry — aggregate into a feature vector, `GaussianNaiveBayes` fits from a short user-labeled calibration session, and the watch predicts the activity type during the workout. See <doc:Naive-Bayes> for the probabilistic framing of how the per-class mean and variance score a new feature vector. The fit-on-the-wrist shape captures a different personal artifact for each discipline: an HR-vs-power curve, a walk-vs-hike-vs-climb classifier, a stroke-type model. Each one persists to Documents as a `Codable` value the next session can decode unchanged.

> Note: Closed-form methods on small per-user data are what the watch is for. Iterative-convergence methods over bulk data — `KMeans` with high `maxIterations`, `KNearestNeighbors` over thousands of training points, `Matrix.invert` on large dimensions — are not. The 30-second watchOS timer budget and the watch FPU's precision profile can return non-converged results that look correct. Move iterative-bulk fitting to iOS or the server.

## Where to go from here

The three sections above each have a deeper layer of math underneath them, and that math is the next step for watchOS developers moving into numerical work. <doc:Statistics-Primer> builds the vocabulary of variance, distributions, and the empirical rule that the baseline section leans on. <doc:Linear-Algebra-Primer> extends vectors and dot products into the geometric operations that read sensor data as orientation and alignment. <doc:Physics-Primitives-Primer> covers the signal-processing surface — integrals, derivatives, and frequency content — that turns short sensor windows into physical quantities. <doc:Machine-Learning-Primer> closes the loop with features, labels, training, and the trade-offs that decide which closed-form method to reach for on the wrist.

> Experiment: **The Quiver Notebook** is the right place to feel how `empiricalRule()` reads against personal baselines. Load a vector of recent HR samples, compare the observed within-1σ fraction to the theoretical 68.27%, and watch the gap shift as the workout type changes. The same `Codable` model that fits in the Notebook decodes unchanged on the watch. See <doc:Quiver-Notebook>.
