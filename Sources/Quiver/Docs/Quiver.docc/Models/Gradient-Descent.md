# Gradient Descent

Fit a regression model by adjusting its coefficients one small step at a time.

## Overview

Think of a ball placed in a bowl — gravity rolls it to the bottom. `GradientDescent` does the same with math instead of gravity. It finds the best coefficients for a linear model by repeating a simple process: check how wrong the current predictions are, figure out which way to adjust each coefficient to make them less wrong, then move each one a small amount in that direction. After enough repetitions, the error stops falling and the model has settled on its answer — the bottom of the bowl.

This is the same answer `LinearRegression` produces in a single matrix expression, only reached step by step instead of computed directly. Both routes converge to the same coefficients on a squared-error problem. The iterative route exists for the regression models that follow this one — logistic regression first — where the error formula has no closed-form answer and stepping toward the minimum is the only way to find it. The fitted model carries every step of the descent as a stored property, so we can see the error fall across iterations, confirm the run converged, and diagnose a run that crawled or diverged.

### How it works

The optimizer repeats the same loop until the error stops falling: start with all coefficients set to zero, compute the current **loss** (how wrong the predictions are — `GradientDescent` uses mean squared error, the same calculation available directly on any prediction array as `meanSquaredError(actual:)`), compute the **gradient** (one number for each coefficient saying which way the loss is heading if that coefficient changes), move each coefficient a small amount in the direction that reduces the loss (the size of that move is the **learning rate**), and repeat — until the loss stops falling by more than a small relative amount, the **convergence tolerance**. When the model converges, additional iterations do not reduce the loss further; the coefficients have settled on the values that nearly minimize the error, and the descent is complete.

The one-coefficient case shows the idea on numbers we can check by hand. The error formula is a bowl with its lowest point at `x = 3`, and the derivative of that formula tells us, at any point, which way the bowl is sloping:

> Note: `Polynomial` stores coefficients in ascending order — `[a₀, a₁, a₂]` represents `a₀ + a₁x + a₂x²`. The first element is the constant term, the second multiplies `x`, the third multiplies `x²`, and so on. See <doc:Polynomials> for the full type.

```swift
import Quiver

// A bowl-shaped error formula: lowest at x = 3.
let error = Polynomial([9.0, -6.0, 1.0])    // x² − 6x + 9
let slope = error.derivative()              // 2x − 6

error(0.0)   // 9.0  — the error if we start with x = 0
slope(0.0)   // -6.0 — slope is negative, so the error falls if we move x to the right

// Take one small step in the downhill direction.
let learningRate = 0.1
let next = 0.0 - learningRate * slope(0.0)  // 0.6
error(next)  // 5.76 — the error after one step
```

We started at `x = 0` with an error of `9.0`. The derivative said "the ground falls if you move right." We moved a small amount to the right; the error dropped to `5.76`. Repeating that move — recheck the slope, take another small step — walks `x` toward `3.0`, where the error is zero and the slope is flat. A real model has many coefficients instead of one, and the error formula is shaped by training data rather than written by hand, but the loop is the same — every coefficient gets its own slope, and every coefficient moves a small step in its own downhill direction.

The optimizer is **batch** and **deterministic**. Batch means every training sample contributes to every step. Deterministic means the same inputs always produce the same trajectory, with no randomness. The learning rate is constant for the whole run; it does not decay, and no penalty term is added to the loss. Variants of gradient descent exist that do those things; keeping the version here simple is what lets the trajectory be readable as a teaching artifact.

### Fitting a model

The `fit(features:targets:learningRate:maxIterations:tolerance:intercept:)` static method runs the descent loop and returns a ready-to-use model. There is no separate unfitted state — the returned struct is immediately usable.

```swift
import Quiver

// Standardize features first — see "Standardize features" below for why this matters.
let features = scaled                       // standardized [[Double]] matrix
let targets  = [3.0, 5.0, 7.0, 9.0, 11.0]

let model = try GradientDescent.fit(features: features, targets: targets)
print(model)
// GradientDescent: 1 feature, converged in 142 iterations (loss: 0.0034)
```

The defaults — `learningRate: 0.01`, `maxIterations: 1000`, `tolerance: 1.0e-6` — are calibrated for standardized features and produce convergence on most well-conditioned problems in well under the iteration cap.

### Watching the descent

Every fitted model carries the loss at every iteration in `lossHistory`. The first entry is the loss at θ = 0; the last entry is the loss at the returned coefficients. Inspecting the trajectory turns "did it converge" from a yes-or-no into a picture:

```swift
import Quiver

let model = try GradientDescent.fit(features: scaled, targets: targets)

model.lossHistory.first    // 57.0  — loss at θ = 0
model.lossHistory.last     // 0.0034 — loss at the returned coefficients
model.iterations           // 142
model.outcome              // .converged
```

A descent that worked falls steeply at first, flattens as it approaches the minimum, and stops when the relative drop falls below `tolerance`. A descent that crawled barely moves and hits the iteration cap. A descent that diverged throws before returning a model at all. The three outcomes are distinguishable without reading any numbers — but the numbers are there when needed.

> Tip: The convergence tolerance is tighter than the value common in ecosystem peers — deliberately so. The trajectory ends when it has actually stopped moving, not when it has nearly stopped. For teaching purposes, "converged" should mean converged.

### Choosing a learning rate

The `learningRate` parameter determines how big each step is. Each step moves every coefficient a small amount in the direction that reduces the loss, and the size of that move is the learning rate. Choosing too small a rate makes the descent crawl — every step covers so little ground that the iteration cap arrives before the minimum does. Choosing too large a rate makes the descent overshoot — each step jumps past the minimum to a worse value, until the loss diverges to infinity.

The right learning rate depends on the data. A common approach is a **learning rate sweep** — fit models across a range of rates and read the outcome off each. Quiver makes the diagnostic visible: `outcome` tells us whether the run converged or hit the cap, `iterations` tells us how quickly, and a divergent rate throws `GradientDescentError`:

```swift
import Quiver

// Sweep four learning rates on the same data.
for rate in [0.0001, 0.01, 0.1, 5.0] {
    do {
        let model = try GradientDescent.fit(features: scaled, targets: targets,
                                             learningRate: rate, maxIterations: 1000)
        print("rate=\(rate): \(model.outcome), \(model.iterations) iterations")
    } catch let error as GradientDescentError {
        print("rate=\(rate): diverged — \(error)")
    }
}
// rate=0.0001: maxIterationsReached, 1000 iterations    ← crawl (rate too small)
// rate=0.01:   converged,             142 iterations    ← healthy
// rate=0.1:    converged,             23 iterations     ← healthy, faster
// rate=5.0:    diverged — …                              ← overshoot (rate too large)
```

Four runs, three outcomes — crawl, converge, diverge. The healthy band is where the descent converges in a reasonable number of iterations; rates above it diverge, rates below it crawl. The diagnostic story replaces guessing.

> Important: An `.maxIterationsReached` outcome is necessary but not sufficient for trustworthiness. A run that hit the cap and made meaningful progress is a useful best-so-far estimate; a run that hit the cap and barely moved is essentially the initial θ = 0 with a rounding error. Compare `lossHistory.first` to `lossHistory.last` to confirm meaningful descent before relying on the coefficients.

### Standardize features

The defaults assume features have already been standardized — typically via `StandardScaler`. On raw-scale features the curvature of the loss surface grows with the squared feature magnitude, and the default learning rate produces an update vastly larger than the distance to the minimum. The descent diverges immediately. The optimizer surfaces this as `GradientDescentError.divergedNonFinite` rather than returning a model with `NaN` coefficients, but the right fix is to scale the inputs:

```swift
import Quiver

let scaler = StandardScaler.fit(features: rawFeatures)
let scaled = scaler.transform(rawFeatures)

let model = try GradientDescent.fit(features: scaled, targets: targets)
```

The same composition works through `Pipeline` when scaling and fitting are kept together as one unit. See <doc:Feature-Scaling> and <doc:Pipeline>.

### Comparing to the closed form

On problems where both routes converge, the iterative answer and the closed-form answer land in the same place — that is the verification that the descent worked. Running the two side by side on standardized data is the cleanest way to confirm the optimizer's mechanics, and the cleanest cookbook story for a reader encountering iterative optimization for the first time.

```swift
import Quiver

let iterative = try GradientDescent.fit(features: scaled, targets: targets)
let closedForm = try LinearRegression.fit(features: scaled, targets: targets)

iterative.coefficients   // ≈ closedForm.coefficients
closedForm.coefficients  // exact answer in one step
```

The two answers agree because gradient descent converges to the same minimum the normal equation solves analytically. The iterative route exists for the case the closed form cannot solve: the next regression model, logistic regression, has a loss function with no closed form. The optimizer that found the linear answer here is the one that will find the logistic answer there, where no shortcut exists.

### When to use which

`LinearRegression` is the right choice for ordinary least squares — it is closed-form, exact, and one pass. Reach for `GradientDescent` for three reasons: when the loss function has no closed form (logistic regression and beyond), when the feature count is large enough that matrix inversion becomes expensive, or when the descent itself is the lesson — when watching the loss fall is the point. For the first two reasons the choice is forced. For the third it is a teaching choice, and the optimizer's first-class trajectory is what makes it the right one.

### Safe by design

The optimizer is built so that the only models the caller sees are trustworthy ones. Three guarantees enforce that contract.

**Divergence throws rather than returning corrupted coefficients.** When the descent overshoots into non-finite loss, or the loss strictly increases beyond the convergence band between iterations, `fit` throws `GradientDescentError`. The alternative — returning a model with `NaN` or `±∞` parameters — would let corrupted numbers propagate silently into prediction pipelines and into downstream models that share this optimizer.

**The descent is observable, not implied.** `lossHistory` carries the loss at every iteration as a stored property. A reader can see whether the descent fell smoothly, plateaued, or stalled, without instrumenting the fit call. The selling point of an iterative optimizer over a black-box one is exactly this visibility.

**Cap-reached is distinguished from convergence.** The `Outcome` enum is a typed value the type system makes the caller acknowledge. A run that hit the iteration cap is not silently labeled "successful"; it is labeled `.maxIterationsReached`, and the caller's downstream code must handle the two cases on purpose.

### From iterative to non-linear

Gradient descent on squared error is the simplest case — the loss is convex, the minimum is unique, and the closed form is available as a check. The next regression problem in the sequence, logistic regression, replaces squared error with log loss to predict probabilities rather than continuous values. Log loss is also convex, but it has no closed-form minimum. The same descent loop, the same learning rate, the same convergence test — applied to a different loss — will be how that model is fit. See <doc:Activation-Functions> for the `sigmoid` function that will sit at the center of that next step.

> Experiment: **The Quiver Notebook** is the right place to feel the learning-rate cliff. Pick a small standardized dataset, then sweep `learningRate` from `0.001` through `0.5` in steps and print `outcome`, `iterations`, and `lossHistory.last` for each run. Watching the trajectory shorten as the rate grows, then watching it explode into `divergedIncreasing` past the threshold, is what makes the optimizer's failure mode concrete. See <doc:Quiver-Notebook>.

## Topics

### Model
- ``GradientDescent``

### Diagnostics
- ``GradientDescent/lossHistory``
- ``GradientDescent/iterations``
- ``GradientDescent/finalLoss``
- ``GradientDescent/outcome``
- ``GradientDescent/Outcome``

### Errors
- ``GradientDescentError``

### Related
- <doc:Feature-Scaling>
- <doc:Pipeline>
- <doc:Calculus-Primer>
- <doc:Machine-Learning-Primer>
