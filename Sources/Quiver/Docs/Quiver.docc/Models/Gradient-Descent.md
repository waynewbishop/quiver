# Gradient Descent

Fit a regression model by adjusting its coefficients one small step at a time.

## Overview

We often find that the error formulas for our models are too complex for a direct algebraic solution. We use **gradient descent** as an iterative tool to walk toward the lowest possible error. This optimizer acts as a shared engine that powers every iterative model in Quiver.

 We start with a guess for our coefficients. We then check how wrong those predictions are and figure out which way to adjust each coefficient to make the model less wrong. We repeat this check and adjustment many times until the error stops falling. The final state represents the best predictions the model can make. 

 This engine is what links our different models together. Whether we are fitting a squared error model or a cross-entropy model we use the same descent loop. This consistency ensures that our convergence tests and divergence guards behave identically across the entire library. This shared design allows us to diagnose why a model might fail and ensures that every model we build benefits from the same numerical safety rules.

### How it works

`GradientDescent` walks an error formula to its minimum using the derivative. The mechanics of why this works — what a derivative tells us, why iterative optimization exists, why the closed-form normal equation is a special case — are covered in <doc:Calculus-Primer>. This section names the specific implementation choices Quiver makes.

The optimizer repeats one loop: compute the current **loss** (Quiver uses mean squared error, available directly as `meanSquaredError(actual:)`), compute the **gradient** (one number per coefficient saying which way the loss is heading if that coefficient changes), move each coefficient a small amount in the downhill direction (the size of the move is the **learning rate**), and repeat until the loss stops falling by more than a small relative amount (the **convergence tolerance**).

The one-coefficient case shows the loop on numbers we can check by hand:

```swift
import Quiver

let error = Polynomial([9.0, -6.0, 1.0])    // x² − 6x + 9, minimum at x = 3
let slope = error.derivative()              // 2x − 6

error(0.0)   // 9.0  — the error if we start with x = 0
slope(0.0)   // -6.0 — slope is negative, so the error falls if we move x to the right

let learningRate = 0.1
let next = 0.0 - learningRate * slope(0.0)  // 0.6
error(next)  // 5.76 — the error after one step
```

Repeating the move — recheck the slope, take another step — walks `x` toward `3.0`, where the error is zero and the slope is flat. A real model has many coefficients and the error formula is shaped by training data rather than written by hand, but the loop is the same: every coefficient gets its own slope, every coefficient moves a small step in its own downhill direction.

For readers who think in algorithms, gradient descent is a greedy local search over a continuous surface. At each position it makes the locally best move — the steepest step downhill — using the derivative to choose the direction instead of trying neighbors one by one. Like any greedy search, it commits: it follows a single line of descent rather than exploring branches, and it never backtracks, which is what makes it fast. The same greediness is also its limit. On a surface with more than one valley it settles into whichever valley it started nearest, the way a depth-first search can return the first goal it reaches rather than the best one — it has no way to climb back out and check the others. This is why the models built on it use error formulas that are convex, a single bowl with one lowest point, where the nearest valley is always the right one.

> Note: The search algorithms this comparison draws on are covered in *Swift Algorithms & Data Structures*: [depth-first search](https://waynewbishop.github.io/swift-algorithms/13-graphs.html) in the Graphs chapter, and the [greedy approach](https://waynewbishop.github.io/swift-algorithms/16-shortest-paths.html) in the Shortest Paths chapter.

The optimizer is **batch** and **deterministic**. Batch means every training sample contributes to every step — which keeps each step exact, and also ties the cost of a step to the size of the dataset, so the variants that trade exactness for sampling a subset per step are the ones reached for when the data grows too large to revisit in full. Deterministic means the same inputs always produce the same trajectory, with no randomness. The learning rate is constant for the whole run; it does not decay, and no penalty term is added to the loss. Variants of gradient descent exist that do those things; keeping the version here simple is what lets the trajectory be readable as a teaching artifact.

### Fitting a model

The `fit(features:targets:learningRate:maxIterations:tolerance:intercept:)` static method runs the descent loop and returns a ready-to-use model. There is no separate unfitted state — the returned struct is immediately usable.

```swift
import Quiver

// Standardize features first — see "Standardize features" below for why this matters.
let rawFeatures: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
let targets = [3.0, 5.0, 7.0, 9.0, 11.0]
let scaler = StandardScaler.fit(features: rawFeatures)
let scaled = scaler.transform(rawFeatures)

let model = try GradientDescent.fit(features: scaled, targets: targets, learningRate: 0.1)
print(model)
// GradientDescent: 1 feature, converged in 101 iterations (loss: 0.0000)
```

The defaults — `learningRate: 0.01`, `maxIterations: 1000`, `tolerance: 1.0e-6` — are deliberately conservative. A small learning rate is the safe choice because it is unlikely to overshoot the minimum, but on a given dataset it may need more than the default cap of iterations to converge; a larger rate, like the `0.1` above, often reaches the minimum far sooner. Choosing the rate is the subject of the next sections.

The `intercept` parameter defaults to `true`, which fits a bias term — the model prepends a column of ones to the features internally, so callers pass their raw feature rows unchanged. When an intercept is fit, `coefficients[0]` is the bias and the remaining entries are the feature weights; the `predict(_:)` method accounts for it automatically. Pass `intercept: false` to force the fit through the origin.

### Watching the descent

Every fitted model carries the loss at every iteration in `lossHistory`. The first entry is the loss at θ = 0; the last entry is the loss at the returned coefficients. Inspecting the trajectory turns "did it converge" from a yes-or-no into a picture. Continuing with the `model` fit above:

```swift
model.lossHistory.first    // 57.0     — loss at θ = 0
model.lossHistory.last     // ~1.5e-18 — effectively zero at the returned coefficients
model.iterations           // 101
model.outcome              // .converged
```

A descent that worked falls steeply at first, flattens as it approaches the minimum, and stops when the relative drop falls below `tolerance`. A descent that crawled barely moves and hits the iteration cap. A descent that diverged throws before returning a model at all. The three outcomes are distinguishable without reading any numbers — but the numbers are there when needed.

> Tip: The convergence tolerance is deliberately tight. The trajectory ends when it has actually stopped moving, not when it has nearly stopped. For teaching purposes, "converged" should mean converged.

### Making predictions

`predict(_:)` returns continuous values for one or more samples. New rows must be scaled with the same `scaler` the model trained on, never a fresh one fit on the new data:

```swift
let newHouses: [[Double]] = [[6.0], [7.0]]
let predictions = model.predict(scaler.transform(newHouses))
// [13.0, 15.0]
```

The model carries no scaler of its own — scaling is the caller's step, kept explicit so the training and prediction transforms cannot silently drift apart. See <doc:Feature-Scaling> and <doc:Pipeline> for bundling the two together.

### Choosing a learning rate

The `learningRate` parameter determines how big each step is. Each step moves every coefficient a small amount in the direction that reduces the loss, and the size of that move is the learning rate. Choosing too small a rate makes the descent crawl — every step covers so little ground that the iteration cap arrives before the minimum does. Choosing too large a rate makes the descent overshoot — each step jumps past the minimum to a worse value, until the loss diverges to infinity.

The right learning rate depends on the data. A common approach is a **learning rate sweep** — fit models across a range of rates and read the outcome off each. Quiver makes the diagnostic visible: `outcome` tells us whether the run converged or hit the cap, `iterations` tells us how quickly, and a divergent rate throws `GradientDescentError`:

```swift
// Sweep four learning rates on the same scaled features.
for rate in [0.0001, 0.1, 0.5, 5.0] {
    do {
        let model = try GradientDescent.fit(features: scaled, targets: targets,
                                             learningRate: rate, maxIterations: 1000)
        print("rate=\(rate): \(model.outcome), \(model.iterations) iterations")
    } catch let error as GradientDescentError {
        print("rate=\(rate): diverged — \(error)")
    }
}
// rate=0.0001: maxIterationsReached, 1000 iterations    ← crawl (rate too small)
// rate=0.1:    converged,             101 iterations    ← healthy
// rate=0.5:    converged,             2 iterations      ← healthy, much faster
// rate=5.0:    diverged — …                              ← overshoot (rate too large)
```

Four runs, three outcomes — crawl, converge, diverge. The healthy band is where the descent converges in a reasonable number of iterations; rates above it diverge, rates below it crawl. On this dataset the crawl-to-converge boundary sits between `0.0001` and `0.1`, which is why the conservative default needs the full iteration budget here while `0.1` finishes in a hundred steps. The diagnostic story replaces guessing.

> Important: An `.maxIterationsReached` outcome is necessary but not sufficient for trustworthiness. A run that hit the cap and made meaningful progress is a useful best-so-far estimate; a run that hit the cap and barely moved is essentially the initial θ = 0 with a rounding error. Compare `lossHistory.first` to `lossHistory.last` to confirm meaningful descent before relying on the coefficients.

### Standardize features

The defaults assume features have already been standardized — typically via `StandardScaler`. On raw-scale features the curvature of the loss surface grows with the squared feature magnitude, and the default learning rate produces an update vastly larger than the distance to the minimum. The descent diverges immediately. The optimizer surfaces this as `GradientDescentError.divergedNonFinite` rather than returning a model with `NaN` coefficients, but the right fix is to scale the inputs:

```swift
import Quiver

let rawFeatures: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
let targets = [3.0, 5.0, 7.0, 9.0, 11.0]

// Fit the scaler on the training features, then transform before fitting.
let scaler = StandardScaler.fit(features: rawFeatures)
let scaled = scaler.transform(rawFeatures)

let model = try GradientDescent.fit(features: scaled, targets: targets, learningRate: 0.1)
```

The same composition works through `Pipeline` when scaling and fitting are kept together as one unit. See <doc:Feature-Scaling> and <doc:Pipeline>.

### Comparing to the closed form

On problems where both routes converge, the iterative answer and the closed-form answer land in the same place — that is the verification that the descent worked. Running the two side by side on standardized data is the cleanest way to confirm the optimizer's mechanics, and the cleanest cookbook story for a reader encountering iterative optimization for the first time.

```swift
// Continuing with the scaled features and targets from above.
let iterative = try GradientDescent.fit(features: scaled, targets: targets, learningRate: 0.1)
let closedForm = try LinearRegression.fit(features: scaled, targets: targets)

iterative.coefficients   // [7.0, 2.83] — reached step by step
closedForm.coefficients  // [7.0, 2.83] — the same answer in one pass
```

The two answers agree because gradient descent converges to the same minimum the normal equation solves analytically. The iterative route exists for the case the closed form cannot solve: <doc:Logistic-Regression> has a cross-entropy loss with no closed form. The optimizer that found the linear answer here is the one that finds the logistic answer there, where no shortcut exists.

### When to use which

`LinearRegression` is the right choice for ordinary least squares — it is closed-form, exact, and one pass. The closed form solves the normal equation in O(*n*·*f*² + *f*³) time, where *n* is the number of samples and *f* the number of features: the *f*³ term is the cost of inverting the *f*×*f* matrix XᵀX, and it is exact but grows quickly as the feature count rises.

`GradientDescent` takes O(*k*·*n*·*f*) time — *k* iterations, each one pass over the *n* samples and *f* features to form the gradient Xᵀ(Xθ − y). Each step is linear in the feature count rather than cubic, so as *f* grows the per-step cost rises far more slowly than the closed-form inversion; the trade is that the descent pays for *k* of those steps and must converge. Where the two routes break even depends on all three of *n*, *f*, and *k* together, not on the feature count alone.

Reach for `GradientDescent` for three reasons. When the loss has no closed form — logistic regression and the margin classifiers beyond it — the iterative route is the only route. When the feature count is large enough that the *f*³ inversion dominates, stepping with a per-iteration cost linear in *f* scales better. And when the descent itself is the lesson — when watching the loss fall is the point — its first-class trajectory is what makes it the right teaching choice. For the first reason the choice is forced; for the other two it is a judgment the two complexities above make possible on a given dataset.

### Safe by design

The optimizer is built so that the failures it can detect are surfaced rather than hidden, and three guarantees enforce that contract. One failure it cannot detect is named at the end.

Divergence throws rather than returning corrupted coefficients. When the descent overshoots into non-finite loss, or the loss strictly increases beyond the convergence band between iterations, `fit` throws `GradientDescentError`. The alternative — returning a model with `NaN` or `±∞` parameters — would let corrupted numbers propagate silently into prediction pipelines and into downstream models that share this optimizer.

The descent is observable, not implied. `lossHistory` carries the loss at every iteration as a stored property, so a reader can see whether the descent fell smoothly, plateaued, or stalled without instrumenting the fit call. The selling point of an iterative optimizer over a black-box one is exactly this visibility.

Cap-reached is distinguished from convergence. The `Outcome` enum is a typed value the type system makes the caller acknowledge. A run that hit the iteration cap is not silently labeled "successful"; it is labeled `.maxIterationsReached`, and the caller's downstream code must handle the two cases on purpose.

One failure these guarantees do not catch is non-identifiable coefficients. When two features carry nearly the same information, infinitely many coefficient vectors fit the data equally well, and the descent converges to one of them — reporting `.converged` with a clean loss and no warning. The predictions are sound, but the individual coefficients are arbitrary: a different starting point would yield different weights with identical predictions. Where the closed-form <doc:Linear-Regression> throws on perfectly collinear features, gradient descent succeeds quietly onto an answer the data cannot justify. The <doc:Regularization-Primer> covers how a penalty resolves this, and <doc:Ridge-Regression> is the model that applies it. For recognizing this failure signature directly — telling a quietly arbitrary fit apart from a trustworthy one — see <doc:Model-Interpretation-Primer>.

### From iterative to non-linear

Gradient descent on squared error is the simplest case — the loss is convex, the minimum is unique, and the closed form is available as a check. The same optimizer also powers regularized regression: <doc:Ridge-Regression> adds a penalty on coefficient size to this same descent, trading a little training accuracy for stability on collinear data, and the <doc:Regularization-Primer> covers when and why to reach for it. <doc:Logistic-Regression> takes the step further from continuous prediction: it replaces squared error with log loss to predict probabilities rather than values. Log loss is also convex, but it has no closed-form minimum — so the same descent loop, the same learning rate, the same convergence test, applied to a cross-entropy loss, is exactly how that model is fit. See <doc:Activation-Functions> for the `sigmoid` function at the center of it, and the <doc:Optimization-Primer> for why one descent algorithm serves several models at once.

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
- ``GradientDescent/learningRate``

### Errors
- ``GradientDescentError``

### Related
- <doc:Feature-Scaling>
- <doc:Pipeline>
- <doc:Calculus-Primer>
- <doc:Regularization-Primer>
- <doc:Machine-Learning-Primer>
