# Regularization Primer

Curbing overfitting by adding a penalty that shrinks coefficients toward zero.

## Overview

A model is **overfitted** when it memorizes the quirks of its training data instead of the underlying pattern. Overall, it scores well on training data but poorly on unseen data. Detecting this gap is a matter of comparing training and test scores; fixing it is the job of **regularization**. This adds a small penalty that discourages a model from leaning too hard on any one feature, ensuring it follows the real pattern rather than the noise. We trade a little accuracy on the training set for steadier, more generalized predictions.

> Note: This primer builds on the overfitting and generalization concepts from the <doc:Machine-Learning-Primer>. For the diagnostic side—recognizing lopsided coefficients before reaching for a cure—see the Model Interpretation Primer.

### When a fit is too good

Overfitting is easiest to see when two features carry nearly the same information. Suppose a dataset records floor area twice—once in square feet and once in square meters. Because they are measured slightly differently, the columns move together almost perfectly, but not exactly.

```swift
import Quiver

// Eight houses, with floor area recorded two ways: square feet and
// square meters. These columns move together almost perfectly.
let features: [[Double]] = [
    [1100, 102], [1400, 130], [1600, 149], [1850, 172],
    [2100, 195], [2400, 223], [2750, 256], [3000, 279]
]
let prices = [205000.0, 245000, 279000, 311000, 360000, 405000, 455000, 498000]
```

Ordinary least squares has no reason to prefer one column over the other, so it can assign one a large positive weight and the other a large compensating negative weight, balancing both on a knife's edge. This instability appears directly in the coefficients. Standardizing features and fitting without a penalty produces a wildly lopsided pair:

```swift
import Quiver

let scaler = StandardScaler.fit(features: features)
let scaled = scaler.transform(features)

let ols = try LinearRegression.fit(features: scaled, targets: prices)
ols.coefficients  // [344750, 608719, -512315] — intercept, then two huge opposite weights
```

These numbers are not findings about floor area; they are the model chasing quirks across two near-identical columns. A different sample would swing them just as violently. The Machine Learning Primer names this failure mode; regularization is our standard response.

### When three fits diverge

The same near-identical columns illustrate how a model's fitting process determines how it handles ambiguous data. The floor-area features push every regressor toward the same wall, but each meets it differently.

Ordinary least squares, shown above, yields a lopsided pair. If we push the collinearity further—two columns that are exact multiples—the matrix inversion fails, and `LinearRegression` throws `MatrixError.singular`. This refusal is honest: with no unique solution, the model provides none.

`GradientDescent` takes a quieter approach. Since it avoids matrix inversion and only walks downhill on the error, it converges on near-collinear data without complaint:

```swift
import Quiver

// The same scaled floor-area features, fit by gradient descent instead.
let descent = try GradientDescent.fit(features: scaled, targets: prices, learningRate: 0.1)
descent.coefficients  // [344747, 48231, 48173] — converged, two near-equal weights
descent.outcome       // .converged
```

The weights return near-equal and far smaller than the least-squares pair, and the outcome reports `.converged` with no warning. This looks like a better answer—and for predictions, it is—but the individual coefficients are not trustworthy. When two columns carry nearly identical information, the valley of equally good fits is so shallow that gradient descent stops somewhere along it, depending entirely on where it started. Begin the descent from a different point, and the two weights land elsewhere. If the columns become perfectly identical, the valley floor goes truly flat; infinitely many weight pairs fit identically, there is no single bottom, and the model predicts while its coefficients remain arbitrary.

This is the trap the penalty closes. Least squares fails loudly, gradient descent succeeds quietly onto an arbitrary answer, and neither has settled on which coefficients to believe. Regularization turns this flat valley of equally good answers into a single, defensible solution.

### The penalty

Regularization changes the objective. While ordinary least squares minimizes the squared error alone, Ridge regression adds a penalty proportional to the squared size of the weights:

```
minimize   (1/n)‖Xθ − y‖²  +  λ‖θ‖²
```

The first term rewards fitting the data; the second punishes large weights. The dial **`lambda`** sets the penalty's strength. At a `lambda` of zero, the penalty vanishes, reproducing ordinary least squares. As `lambda` grows, large weights become expensive, forcing the model to keep them small unless the data strongly insists otherwise. We leave the intercept out of the penalty—only slopes are asked to shrink, as the baseline value is not what overfits.

### Watching coefficients shrink

The penalty's effect is tangible. Sweeping `lambda` across the floor-area data—the same columns that yielded the wild `608719` and `-512315` weights—shows the coefficients contracting and stabilizing:

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

As `lambda` rises, two things happen. First, weights shrink; the pair that ranged over a million apart at no penalty drops to near `32000` at `lambda` of one, and near `8000` at `lambda` of ten. Second, the weights converge, the penalty spreading the load evenly across the near-identical columns rather than letting one swing wildly positive and the other negative. The intercept stays near `344500` throughout since it is never penalized. The lopsided fit becomes a modest, balanced one. See Ridge Regression for the model itself.

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

The penalty that shrinks coefficients is the same penalty that makes the arithmetic tractable. Quiver's `Ridge` reaches the same minimum using gradient descent on the penalized objective, rather than inverting this matrix directly, but the stabilizing effect remains identical. Both routes arrive at the same minimum; our choice is a tradeoff between closed-form exactness and descent's scalability for larger problems.

> Tip: The Determinants Primer introduces `conditionNumber` as the diagnostic that flags a near-singular matrix. Regularization is the response that diagnostic points toward.

### Choosing the penalty strength

This dial requires a gauge. A small `lambda` leaves the model free to overfit; a large `lambda` shrinks weights so far the model underfits—too rigid to follow the real pattern. Overfitting and underfitting are not separate problems to solve; they are opposite ends of one dial. The art lies not in eliminating failure, but in finding the setting where the model generalizes best.

We do not guess this setting; we measure it. Training error is an unreliable guide, as it always falls whenever `lambda` falls, nominating a zero penalty—the very overfit we want to cure. The honest gauge is performance on held-out data. **Cross-validation** supplies this: we split data into folds, fit on most, score on the hold-out, and rotate so every fold takes a turn. The `lambda` with the best average score across folds is a defensible choice rather than an artifact of training data. See Train-Test Split for the mechanics of this evaluation.

> Important: Training error cannot choose `lambda`. It falls as `lambda` falls, so it always points back to no penalty at all. Choose `lambda` by measuring performance on held-out data—the cross-validated score—not by reading error on data the model already used for fitting.

### Where regularization goes from here

Ridge shrinks coefficients toward zero but never reaches it—the squared penalty eases off as a weight approaches zero, allowing small weights to survive. A different penalty, proportional to the absolute size of weights, drives the smallest coefficients exactly to zero, performing feature selection as a side effect. Both belong to the same family: adding a penalty to a loss function to buy stability on unseen data.

This concept is foundational to machine learning. Whenever a problem has too little data to pin down a single trustworthy answer—an **ill-posed problem**—a small, well-chosen penalty trades a touch of accuracy for a great deal of stability. The full theory of that trade belongs to a course and a textbook. Quiver is the bench where ridge regression stops being abstract and becomes a tool we can fit, sweep, and watch.

> Experiment: **The Quiver Notebook** is the right place to feel the penalty work. Take two nearly identical features, then sweep `lambda` from `0.0` upward, printing the coefficients and the `conditionNumber` of the design at each step. Watching the weights converge and shrink while the condition number falls from the thousands toward single digits is the fastest way to see that the penalty for overfitting and the cure for instability are the same number. See Quiver Notebook.
