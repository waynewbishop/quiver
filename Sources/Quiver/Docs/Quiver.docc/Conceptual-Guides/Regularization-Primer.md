# Regularization Primer

Curbing overfitting by adding a penalty that shrinks coefficients toward zero.

## Overview

A model is **overfitted** when it has memorized the quirks of its training data instead of the real pattern — it scores well on its training data and poorly on data it has never seen. Detecting that gap is the job of comparing training and test scores; fixing it is the job of **regularization** — adding a small penalty that discourages a model from leaning too hard on any one feature, so it follows the real pattern and not the quirks. The penalty trades a little accuracy on the training set for steadier predictions in the world, which is the trade that matters whenever a model must generalize beyond its original training data.

> Note: This primer builds on the overfitting and generalization concepts introduced in the <doc:Machine-Learning-Primer>, and the `conditionNumber` diagnostic from the <doc:Determinants-Primer>. For the diagnostic side — reading the lopsided-coefficient signature off the console and recognizing the failure before reaching for the cure — see <doc:Model-Interpretation-Primer>.

### When a fit is too good

Overfitting is easiest to see when two features carry nearly the same information. Suppose a house dataset records floor area twice — once in square feet, once in square meters — measured slightly differently, so the two columns move together almost perfectly but not exactly:

```swift
import Quiver

// The same eight houses, with floor area recorded two ways: square feet and
// square meters. Each row is [squareFeet, squareMeters] — the same measurement
// in different units, so the two columns move together almost perfectly.
let features: [[Double]] = [
    [1100, 102], [1400, 130], [1600, 149], [1850, 172],
    [2100, 195], [2400, 223], [2750, 256], [3000, 279]
]
let prices = [205000.0, 245000, 279000, 311000, 360000, 405000, 455000, 498000]
```

Ordinary least squares has no reason to prefer one column over the other, so it can hand one a large positive weight and the other a large compensating weight, balancing the two on a knife's edge. The instability shows up directly in the coefficients. Standardizing the features and fitting without any penalty produces a wildly lopsided pair:

```swift
import Quiver

let scaler = StandardScaler.fit(features: features)
let scaled = scaler.transform(features)

let ols = try LinearRegression.fit(features: scaled, targets: prices)
ols.coefficients  // [344750, 608719, -512315] — intercept, then two huge opposite weights
```

The two columns describe the same thing, yet one weight is a large positive number and the other a large negative one that nearly cancels it. That `608719` and `-512315` are not findings about floor area — they are the model chasing quirks across two near-identical columns, and a different sample would swing them just as violently the other way. The <doc:Machine-Learning-Primer> names this failure mode; regularization is the standard response to it.

### When three fits diverge

The same near-identical columns expose a deeper point: how a model is fit decides what it does with data it cannot pin down. The floor-area features push every regressor toward the same wall, and each one meets it differently.

Ordinary least squares, above, hands back the lopsided pair. Push the collinearity all the way — two columns that are exact multiples of each other — and the matrix it inverts becomes singular, and ``LinearRegression`` throws ``MatrixError/singular`` rather than return a number it cannot justify. The refusal is honest: with no unique answer to give, it gives none.

Gradient descent meets the same data and does something quieter. It never inverts a matrix; it only walks downhill on the error. So on the near-collinear floor-area features it converges without complaint:

```swift
import Quiver

// The same scaled floor-area features, fit by gradient descent instead.
let descent = try GradientDescent.fit(features: scaled, targets: prices, learningRate: 0.1)
descent.coefficients  // [344747, 48231, 48173] — converged, two near-equal weights
descent.outcome       // .converged
```

The two weights come back near-equal and far smaller than the least-squares pair, and the outcome reports `.converged` with no warning attached. This looks like the better answer — and for predictions, it is fine: the fitted values are sound. But the individual coefficients are not trustworthy. When two columns carry nearly the same information, the valley of near-equal fits is so shallow that gradient descent stops partway down it — at the point its path from a zero start happens to reach before the loss stops falling by more than the tolerance. Begin the descent from a different point and the two weights land somewhere else entirely, with predictions just as good. Push the columns to perfectly identical and the valley floor goes truly flat — then infinitely many weight pairs fit identically, there is no single bottom at all, and the closed form has nothing to invert. The model predicts; the individual coefficients do not mean anything on their own.

That is the trap the penalty closes. Least squares fails loudly, gradient descent succeeds quietly onto an arbitrary answer, and neither has settled which coefficients to believe. Regularization is what turns the flat valley of equally-good answers into a single defensible one.

### The penalty

Regularization changes what the fit is trying to minimize. Ordinary least squares minimizes the squared error alone. Ridge regression minimizes the squared error plus a penalty proportional to the squared size of the weights:

```
minimize   (1/n)‖Xθ − y‖²  +  λ‖θ‖²
```

The first term rewards fitting the data; the second term punishes large weights. The dial **`lambda`** sets how much the punishment counts. At a `lambda` of zero the penalty vanishes and the fit is ordinary least squares. As `lambda` grows, large weights become expensive, so the fit keeps them small unless the data strongly insists they earn their size. The intercept is left out of the penalty — only the slopes are asked to shrink, since the baseline value is not what overfits.

### Watching coefficients shrink

The penalty's effect is something we can watch rather than take on faith. Sweeping `lambda` across a range on the same floor-area data — the two columns that gave least squares its wild `608719` and `-512315` — shows the weights contracting and settling:

```swift
import Quiver

// The same scaled floor-area features, fit across a range of penalty strengths.
for lambda in [1.0, 10.0] {
    let model = try Ridge.fit(features: scaled, targets: prices, lambda: lambda)
    print(model.coefficients)
}
// no penalty  → [344750, 608719, -512315]   lopsided, enormous
// lambda = 1  → [344480,  32137,   32132]   equal, far smaller
// lambda = 10 → [344320,   8034,    8033]   equal, smaller still
```

Two things change as `lambda` rises. The weights shrink — the pair that ranged over a million apart at no penalty sits near `32000` at a `lambda` of one and near `8000` at a `lambda` of ten. And the two weights converge toward each other, the penalty spreading the work evenly across the near-identical columns rather than letting one swing positive and the other negative. The intercept stays near `344500` throughout, because it is never penalized — only the slopes give ground. The lopsided fit at the top becomes a modest, balanced one below. See <doc:Ridge-Regression> for the model that produces these fits.

### The same penalty that steadies the math

There is a second reason the penalty helps, and it is the same idea seen from the math side. Solving for the coefficients means inverting a matrix built from the features. When two features carry nearly the same information, that matrix is almost impossible to invert cleanly — the same instability that gave us those wild opposite-signed weights. The <doc:Determinants-Primer> measures how close a matrix is to that edge with its `conditionNumber`: a small value is healthy, a huge value is trouble.

Adding the penalty to that matrix nudges it back to safe ground. The effect is direct — for the floor-area data above, the condition number falls from the hundreds of thousands to a handful once the penalty is on the diagonal:

```swift
// Continuing with the scaled floor-area features from above.
// The feature matrix that least squares inverts, before and after the penalty.
let gram = scaled.transposed().multiplyMatrix(scaled)
gram.conditionNumber  // 402610 — almost impossible to invert

// Adding the penalty to the diagonal is what stabilizes the inversion.
var penalized = gram
penalized[0][0] += 1
penalized[1][1] += 1
penalized.conditionNumber  // 17 — the same matrix, made stable
```

The penalty that shrinks the coefficients is the same penalty that makes the matrix tractable. The penalty that keeps the weights honest is the penalty that keeps the arithmetic stable — one dial, two payoffs. Quiver's ``Ridge`` reaches the same minimum by gradient descent on the penalized objective rather than by inverting this matrix directly, but the stabilizing effect of the penalty is exactly the one shown here.

Both routes arrive at the same minimum, and the choice between them is a tradeoff: the closed form is exact and finishes in one pass but must invert a matrix, while descent gives up the one-pass exactness to reuse the shared optimizer and scale to larger problems where forming and inverting that matrix is the expensive step.

> Tip: The <doc:Determinants-Primer> introduces `conditionNumber` as the diagnostic that flags a near-singular matrix. Regularization is the response that diagnostic points toward.

### Choosing the penalty strength

The dial needs a gauge. A small `lambda` leaves the model free to overfit; a large `lambda` shrinks the weights so far the model underfits, too rigid to follow the real pattern. Overfitting and underfitting are not two separate problems to solve one at a time — they are the two ends of a single dial, and `lambda` is the hand that turns it. The art is not eliminating either failure but finding the setting between them where the model generalizes best, and that setting is something we measure rather than guess.

The measurement cannot come from training error. As `lambda` falls, training error only ever falls with it, so training error always nominates a `lambda` of zero — the very overfit we set out to cure. The honest gauge is performance on data the model did not fit. **Cross-validation** supplies it: split the data into folds, fit on most of them, score on the held-out fold, and rotate so every fold takes a turn. The `lambda` with the best average score across folds is a defensible choice rather than an artifact of the data on which the model trained. See <doc:Train-Test-Split> for the held-out evaluation underneath this approach.

> Important: Training error cannot choose `lambda`. It falls as `lambda` falls, so it always points back to no penalty at all. Choose `lambda` by measuring on held-out data — the cross-validated score — not by reading the error on the data the model already used for fitting.

### Where regularization goes from here

Ridge shrinks every coefficient toward zero but never quite to it — the squared penalty eases off as a weight approaches zero, so small weights survive. A different penalty, one proportional to the absolute size of the weights rather than the square, drives the smallest coefficients exactly to zero and selects features as a side effect. Both belong to the same family: the idea of adding a penalty to a loss to buy stability on data the model has not seen.

That idea is older and broader than machine learning. Whenever a problem has too little data to pin down a single trustworthy answer — an **ill-posed problem** — a small, well-chosen penalty trades a touch of accuracy for a great deal of stability. The full theory of that trade belongs to a course and a textbook. Quiver is the bench where one instance of it, ridge regression, stops being abstract and becomes something we can fit, sweep, and watch.

> Experiment: **The Quiver Notebook** is the right place to feel the penalty work. Take two nearly identical features, then sweep `lambda` from `0.0` upward, printing the coefficients and the `conditionNumber` of the design at each step. Watching the weights converge and shrink while the condition number falls from the thousands toward single digits is the fastest way to see that the penalty for overfitting and the cure for instability are the same number. See <doc:Quiver-Notebook>.
