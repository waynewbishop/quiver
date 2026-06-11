# Optimization Primer

Using one training method to fit many different models.

## Overview

Some models compute their answer in a single step. <doc:Linear-Regression> solves a matrix equation based on a defined closed-form formula. Other models are iterative: their coefficients cannot be written as one formula, so the answer has to be searched for rather than solved. We start at some initial point, measure how wrong the guess is, adjust, and repeat until the error stops falling. That iterative sequence is **gradient descent**. 

The useful part is that the search itself changes only slightly from one model to the next. The loop that fits a plain regression is the same loop that fits a penalized one — what differs is the error formula each is handed. This primer is about that shared algorithm: why one optimizer serves several models, what stays fixed and what varies, and where the reuse stops.

> Note: This primer builds on the iterative-minimization mechanics from the <doc:Calculus-Primer> and the optimizer's own model page, <doc:Gradient-Descent>. It assumes the overfitting vocabulary of the <doc:Machine-Learning-Primer>.

### One loop, many losses

The reuse is easiest to see by analogy. A cabinetmaker owns one router. The motor, the bit, and the motion never change — what changes is the **jig**, the fixture that guides the cut along a different path for a different job. One tool, many results, because the guide changes and the tool does not.

Gradient descent works the same way. The loop walks downhill toward the lowest error, asking the same two questions at every step: how wrong is the current guess, and which way should each coefficient move to make it less wrong. The answers come from a **loss function** — the error formula being minimized. Change the loss and the same loop fits a different model. The loop is the router; the loss is the jig — and as with a router, the jig is fixed in place before the cut, not adjusted mid-run.

Three things stay fixed across every model the algorithm fits: the learning rate that sizes each step, the convergence test that decides when to stop, and the divergence guard that throws rather than return corrupted numbers. One thing varies: the loss. That single point of variation is what lets one well-tested algorithm fit a family of models.

The loss is chosen by the model, not passed in as an argument. ``GradientDescent``, ``Ridge``, and ``LogisticRegression`` each carry their own — squared error, squared error plus a penalty, and cross-entropy — fixed at the point the model is defined. There is no separate loss object to supply at the call site; a new loss means a new model, which is why each model below is a distinct type rather than a configuration of one.

### The optimizer that is also a model

It is natural to expect the method that fits a model and the model itself to be two separate things — the algorithm a hidden step, the model the object we hold. Strictly, they are: the model is the straight-line fit and `GradientDescent` is the method that fits it, the same separation every other model on this page keeps. `GradientDescent` is the one case where naming them apart buys nothing. Minimizing squared error over a straight line leaves no separate object to wrap around the optimizer — no squashing function, no penalty term, nothing the algorithm does not already contain. So the type carries the algorithm's name and serves as the model too, not because the two are one thing but because here there is only one thing to name. `GradientDescent` is at once a regression model and the public face of the descent algorithm the penalized models share.

The simplest case stands on its own so it can be watched in isolation — one loss, one bowl, one trajectory to the minimum — before the same algorithm reappears inside `Ridge`, where a penalty term sits on top of it. The plain model is where the method is learned; the others are where it is reused. This single naming is what the simplest case allows. The models that follow keep the same algorithm and wrap a real model around it, so the separation that was always there becomes visible again: in `LogisticRegression` the hypothesis and the optimizer are plainly two things — a sigmoid prediction fitted by the very same descent.

### The models the algorithm fits

`GradientDescent`, the model just described, is the first the algorithm fits: its loss is the squared error, so it fits a straight line by walking to the bottom of a simple bowl one step at a time. Three more models stand in relation to it, and each one teaches its difference by how it departs from that plainest case.

`Ridge` uses the identical loop with one addition to the loss: a penalty proportional to the squared size of the weights. Internally nothing else changes — the same step rule, the same convergence test, the same divergence guard run untouched, and only the error formula they are handed grows one term richer. That single extra term is the whole difference between a plain fit and a regularized one, which is why a reader who can follow `GradientDescent` can already follow `Ridge`. The <doc:Regularization-Primer> covers what the penalty buys.

``LogisticRegression`` departs further. The model keeps the same loop but changes two things at once: the linear output is wrapped in a sigmoid so the model predicts a probability rather than a continuous value, and the squared-error loss gives way to cross-entropy, the loss suited to probabilities. The step rule, convergence test, and divergence guard are still the ones learned here — only the gradient and loss differ. Because cross-entropy has no closed-form minimum, the iterative route is the only route — there is no matrix equation to solve instead. The <doc:Logistic-Regression> page covers the model, and <doc:Activation-Functions> covers the sigmoid at its center.

``LinearRegression`` works the opposite way: it does not use the loop at all. Instead of stepping toward the answer, it solves the squared-error problem in a single matrix equation — a closed form that lands on the minimum directly, with no iterations to watch. On the same data it reaches the same answer `GradientDescent` walks toward, which makes it both a contrast and a check: the one model here that takes a different route to an identical destination.

```swift
import Quiver

// Five samples on a line, standardized so both models receive the same scaled
// data — a like-for-like comparison.
let points: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
let outcomes = [3.0, 5.0, 7.0, 9.0, 11.0]
let scaler = StandardScaler.fit(features: points)
let ready = scaler.transform(points)

// The same data, fit two ways — closed form and iterative descent.
let closed = try LinearRegression.fit(features: ready, targets: outcomes)
let walked = try GradientDescent.fit(features: ready, targets: outcomes, learningRate: 0.1)

closed.coefficients  // [7.0, 2.83] — solved in one matrix step, no trajectory by construction
walked.coefficients  // [7.0, 2.83] — the same answer to displayed precision, walked one step at a time
```

In each coefficient array the first entry is the intercept and the rest are the feature weights, so `[7.0, 2.83]` reads as an intercept of `7.0` and a single slope of `2.83`. The coefficients match because the problem is the same. The difference is what each route leaves behind: the closed form returns a destination and nothing else, because it never traveled, while the descent records every step it took to get there. Reading that record is the skill this primer teaches, and the closed form has no equivalent to read.

```swift
// Only the iterative route leaves a trail of its own descent.
walked.lossHistory[0]   // 57.0    — error at the zero-coefficient start
walked.lossHistory[10]  // 0.6572  — an order of magnitude down in ten steps
walked.lossHistory[20]  // 0.0076  — and again, the bowl flattening toward the floor
walked.iterations       // 101     — steps the closed form never takes
walked.outcome          // .converged — a verdict the closed form has no place for
```

Those three readings are the curve itself, read as numbers rather than drawn: a steep early drop that flattens toward a floor is the shape of a healthy descent. A run that instead climbed from one reading to the next would mark a learning rate too large, and one that inched down by a sliver each step would mark a rate too small. Reading the shape off the milestones is the same judgment a plotted curve invites, made directly from the array.

> Important: `lossHistory` holds one entry per step *plus* the starting point, so its count is `iterations + 1`, not `iterations`. Entry `[0]` is the loss at the zero-coefficient start, before any step is taken, and each later entry is the loss after that step — so a run of `101` iterations leaves a `lossHistory` of `102` values, indexed `0` through `101`. Reading `lossHistory[iterations]` gives the final loss; indexing past it is out of bounds.

Both are recorded when the model is fit, and printing them side by side makes the extra entry visible — the loss array carries one more value than the number of steps, because it also stores the loss before the first step:

```swift
walked.iterations        // 101 — number of steps the descent took
walked.lossHistory.count // 102 — one stored loss per step, plus the loss at the start
```

Because `lossHistory` is an ordinary array, the whole descent can be walked step by step with `enumerated()`:

```swift
// Pair each loss with the step that produced it.
for (step, loss) in walked.lossHistory.enumerated() {
    print("step \(step): \(loss)")
}
// step 0: 57.0   step 1: 36.48   step 2: 23.35   ...   step 101: ~0.0
```

That agreement is the validation: when both routes are available, they land in the same place. The closed form lands there directly; the descent shows the path, and the path is what carries trust to the problems where no closed form exists to check against.

### One skill, carried forward

Because every model the algorithm fits runs the same loop, what a reader learns to watch on the simplest one transfers unchanged to all the rest.

The diagnostics read the same everywhere. Every model the algorithm fits carries its full `lossHistory` and a typed `Outcome`, so reading a convergence trajectory is one skill, learned once on a plain regression and applied without change to a penalized one. The error trajectory of a ridge fit has the same shape, the same flattening toward the minimum, the same `.converged` or `.maxIterationsReached` verdict — because it is the same loop producing it. A divergence looks the same, a crawl looks the same, a healthy descent looks the same, whichever model is being fit.

That transfer is the payoff. A reader who learns to tell a smooth descent from a stalled one, to recognize when the learning rate is too large, to read the loss falling step by step, has learned it for the whole family at once. The next model does not bring a new optimizer to study — it brings a new loss to the optimizer already understood.

### Where one answer becomes many

When two features carry nearly the same information, the error surface stops being a tidy round bowl. The surface stretches into a long, shallow valley — a floor so nearly flat that a wide range of coefficient pairs fit almost equally well, with the single lowest point sitting at the bottom of a trough almost too shallow to detect. Push the features to perfectly collinear and the floor goes truly flat: a whole line of coefficient pairs fit identically and no lowest point exists at all. The closed-form `LinearRegression` needs that lowest point to invert its matrix, and on perfectly collinear features there is none, so it throws ``MatrixError/singular``. The refusal is honest: with no unique answer to give, it gives none.

Gradient descent never inverts anything. The method only walks downhill, so on the near-identical features it descends into the shallow valley, reaches the point where the error stops falling by more than the tolerance, and halts there — reporting `.converged` with a clean loss and no warning. The predictions are sound, but the individual coefficients are arbitrary: they are wherever the walk from a zero start happened to stop in that flat trough, and a different starting point would land elsewhere with predictions just as good. The model predicts; the individual coefficients do not mean anything on their own.

The same data makes the contrast concrete. On the two near-identical floor-area columns from the <doc:Regularization-Primer>, the descent settles onto one point in the shallow valley:

```swift
import Quiver

// Floor area recorded two ways — square feet and square meters — so the two
// columns move together almost perfectly. Each row is [squareFeet, squareMeters].
let floorArea: [[Double]] = [
    [1100, 102], [1400, 130], [1600, 149], [1850, 172],
    [2100, 195], [2400, 223], [2750, 256], [3000, 279]
]
let prices = [205000.0, 245000, 279000, 311000, 360000, 405000, 455000, 498000]

let areaScaler = StandardScaler.fit(features: floorArea)
let areaScaled = areaScaler.transform(floorArea)

// The closed form fits in one step; descent walks the shallow valley instead.
let collinearFit = try GradientDescent.fit(features: areaScaled, targets: prices, learningRate: 0.1)
collinearFit.coefficients  // [344747, 48231, 48173] — the two near-equal weights are the signal split evenly across two columns that carry the same information
collinearFit.outcome       // .converged
```

Those two weights, `48231` and `48173`, are nearly equal for a reason: with both columns carrying the same information, descent split the signal between them rather than committing to either, and any other split would fit the data just as well. The lesson is that the loss decides what the minimum is. With well-separated features the minimum is a point; with collinear features it is a valley, and descent settles somewhere on it silently while the closed form refuses outright.

Because the descent gives no warning, the check has to happen on the features rather than the result. The `conditionNumber` of the design matrix is the diagnostic: a small value means the features are well separated and the coefficients can be trusted individually, while a large value means they are nearly collinear and the individual weights are not. Reading it before trusting the coefficients is how an application catches the silent case the descent will not flag. The <doc:Regularization-Primer> walks through how a penalty tilts that valley back into a single defensible point, the <doc:Determinants-Primer> covers `conditionNumber` itself, and <doc:Ridge-Regression> is the model that applies the penalty. The <doc:Model-Interpretation-Primer> puts this check in context — how to read a fitted model and tell a trustworthy coefficient apart from one the descent left arbitrary.

### Where the algorithm stops

One loop does not fit every model, and naming the boundary is what keeps the claim honest.

The algorithm assumes the loss has a gradient everywhere — a well-defined downhill direction at every point. A penalty proportional to the absolute size of the weights, rather than the square, breaks that assumption: its error surface has sharp creases where the gradient is undefined, so it needs a different kind of optimizer that this loop does not provide. Other models, built around a constrained objective of an entirely different shape, need their own solvers as well. The claim is not that one algorithm fits every model — it is that one algorithm fits the family whose loss is smooth, and that family is large enough to be worth building once.

### From one algorithm to many models

The algorithm introduced here for regression is the one the classification models need too. Each arrives mainly by changing the loss the loop is handed, not by rewriting the loop. ``LogisticRegression`` predicts probabilities by wrapping its linear output in a squashing function and minimizing a loss suited to probabilities rather than squared error — and because that loss has no closed-form answer either, the iterative algorithm is not a convenience but the only route. The same is true of the margin-maximizing classifiers that will follow. The loop a reader learns on a straight line is the loop that fits all of them.

See the <doc:Calculus-Primer> for the derivative that powers every step, <doc:Gradient-Descent> for the optimizer's own page, <doc:Ridge-Regression> for the first model that adds to its loss, and <doc:Logistic-Regression> for the first that changes the prediction as well.

> Experiment: **The Quiver Notebook** is the right place to see the algorithm at work across models. Fit ``GradientDescent``, ``Ridge``, and ``LogisticRegression`` on standardized data, then plot each `lossHistory` against its iteration index and read the curves side by side. Watching three different models produce the same shape of trajectory — same convergence, same diagnostics — from the same loop is what makes the shared algorithm tangible. See <doc:Quiver-Notebook>.
