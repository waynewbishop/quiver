# Polynomials

Represent, evaluate, differentiate, and fit single-variable polynomials.

## Overview

A **polynomial** is an expression of the form `a₀ + a₁x + a₂x² + ... + aₙxⁿ`, a linear combination of powers of a single variable. Polynomials describe trajectories, model curved trends in data, approximate complicated functions on a small interval, and underpin a great deal of numerical computing. Quiver's `Polynomial` type is a value type that stores those coefficients and provides evaluation, arithmetic, differentiation, least-squares fitting, and algebraic rendering in one place.

The coefficients are stored in **ascending order of power**: `coefficients[0]` is the constant term, `coefficients[1]` is the linear term, `coefficients[2]` is the quadratic term, and so on. This matches the convention used in numerical computing libraries that work with polynomials as flat coefficient arrays. The `asExpression` method reverses the order to match human convention, rendering the highest-degree term first.

```swift
import Quiver

// Coefficients ascending: a₀=1, a₁=3, a₂=2
let p = Polynomial([1, 3, 2])

p(2)                    // 15.0  — evaluate at a single point
p([0, 1, 2, 3])         // [1.0, 6.0, 15.0, 28.0] — vectorized
p.derivative()          // 4x + 3
p.asExpression()        // "2x² + 3x + 1"
```

> Tip: For the rendering side of `Polynomial` — how `asExpression` formats the descending-power form, the `relativeZeroTolerance` parameter that suppresses numerical noise in fitted coefficients, and the broader display family on vectors and matrices — see <doc:Rendering-Math-Primer>.

### Evaluating polynomials

A `Polynomial` value is callable. Pass a single `Double` to evaluate at one point, or pass a `[Double]` to evaluate at every point in the array — the second form is the right shape for plotting:

```swift
import Quiver

let p = Polynomial([1, 3, 2])
p.asExpression()        // "2x² + 3x + 1"

p(5)                    // 2·25 + 3·5 + 1 = 66.0
p(-1)                   // 2·1 + 3·(-1) + 1 = 0.0
p(0)                    // 2·0 + 3·0 + 1 = 1.0

let xs = Array.linspace(start: -2.0, end: 2.0, count: 5)
p(xs)                   // [3.0, 0.0, 1.0, 6.0, 15.0]
```

Evaluation is numerically stable — Quiver uses Horner's method internally to avoid the precision loss that comes from repeated `pow` calls. See <doc:Numerical-Literacy> for the broader pattern of reformulating textbook formulas to keep floating-point error small.

### Polynomial arithmetic

Polynomials add, subtract (via scalar negation), and multiply naturally. Addition pads the shorter coefficient array with zeros and adds element-wise; multiplication is the discrete convolution of the two coefficient arrays, where the coefficient at index `k` of the product is `Σ lhs[i] · rhs[k - i]` over all valid `i`. Scalar multiplication scales every coefficient by the same factor:

```swift
import Quiver

// (x + 1)(x - 1) = x² - 1
let p = Polynomial([1, 1])     // x + 1
let q = Polynomial([-1, 1])    // x - 1
p * q                           // Polynomial([-1, 0, 1])  → x² - 1

// Sums fold matching powers together
let r = Polynomial([1, 3, 2])  // 2x² + 3x + 1
let s = Polynomial([4, -3])    // -3x + 4
r + s                           // Polynomial([5, 0, 2])  → 2x² + 5

// Scalar product on the left
3.0 * Polynomial([1, 3, 2])    // Polynomial([3, 9, 6])  → 6x² + 9x + 3
```

### The derivative

The `derivative` method returns the polynomial whose coefficients are obtained by the standard rule of calculus: `d/dx[a₀ + a₁x + a₂x² + ... + aₙxⁿ] = a₁ + 2a₂x + 3a₃x² + ... + n·aₙxⁿ⁻¹`. A constant polynomial differentiates to zero:

```swift
import Quiver

// 2x² + 3x + 1 → 4x + 3
let p = Polynomial([1, 3, 2])
p.derivative()                  // Polynomial([3, 4])

// Evaluating the derivative gives the slope of p at any point
p.derivative()(0)               // 3.0  — slope of p at x = 0
p.derivative()(1)               // 7.0  — slope of p at x = 1

// Constants flatten to zero
Polynomial([5]).derivative()    // Polynomial([0])
```

Because the coefficients are stored in ascending order of power, the rule has a plain-English reading that maps directly onto the array. Each coefficient lives in the slot numbered after its power, so the constant sits at index `0`, the `x` term at index `1`, and the `x²` term at index `2`. Differentiating multiplies each coefficient by its own power and moves it down one slot. The power that did the multiplying is exactly the old index, and the slot it lands in is one position lower — the constant has nowhere lower to go, so it drops out entirely:

```swift
import Quiver

// 2x² + 3x + 1, stored low-to-high as [a₀, a₁, a₂]
let p = Polynomial([1, 3, 2])

// Indexing starts at 0, so each coefficient's index equals its power —
// "multiply by the power" reads as "multiply by the index":
// Index 1 holds 3 (the x term):   3 × 1 = 3, moves to index 0
// Index 2 holds 2 (the x² term):  2 × 2 = 4, moves to index 1
// Index 0 holds 1 (the constant): 1 × 0 = 0, drops out — a constant does not change
p.derivative()                  // Polynomial([3, 4]) → 4x + 3
```

The derivative is a polynomial in its own right, so it composes with everything else: call `derivative` again for the second derivative, evaluate it at a grid of points, add it to another polynomial, or take its own derivative.

### Fitting polynomials to data

The `[Double].polyfit(x:y:degree:)` method is the least-squares counterpart to `LinearRegression.fit`. Given parallel `x` and `y` arrays and a target `degree`, it returns the polynomial of that degree that best fits the data in the ordinary-least-squares sense, the polynomial whose values minimize the sum of squared residuals against `y`. Internally `polyfit` builds a Vandermonde-style design matrix whose row `i` contains `[x[i], x[i]², ..., x[i]ⁿ]`, then defers to `LinearRegression.fit` to solve the normal equation:

```swift
import Quiver

// Underlying truth: 2x² + 3x + 1, evaluated at x = 1...5
let x = [1.0, 2.0, 3.0, 4.0, 5.0]
let y = [6.0, 15.0, 28.0, 45.0, 66.0]

if let p = [Double].polyfit(x: x, y: y, degree: 2) {
    p.coefficients   // [1.0, 3.0, 2.0]  — recovers the original polynomial
    p(6)             // 91.0             — predicted value at a new x
}
```

Because `polyfit` is built on <doc:Linear-Regression>, calling `polyfit(degree: 1)` returns the same line that `LinearRegression.fit(features: x, targets: y)` would. Two doors to the same math. Higher degrees fit curves that linear regression cannot. The function returns `nil` when the inputs are invalid (mismatched lengths, fewer points than `degree + 1`, negative degree) or when the underlying linear system is ill-conditioned.

The equivalence generalizes to any degree. Building the same `[x, x²]` feature matrix by hand and passing it to `LinearRegression.fit` recovers identical coefficients, packaged as an intercept plus weight vector instead of a polynomial:

```swift
// Build the [x, x²] feature matrix. The intercept column is added by fit().
var features: [[Double]] = []
for value in x {
    features.append([value, value * value])
}

let model = try LinearRegression.fit(features: features, targets: y)

model.intercept           // 1.0          — matches polyfit's a₀
model.coefficients        // [3.0, 2.0]   — matches polyfit's [a₁, a₂]
model.predict([[6, 36]])  // [91.0]       — same prediction
```

Choose `LinearRegression.fit` when standard errors, p-values, or confidence intervals on the coefficients matter — `polyfit` does not surface them, and the path to get them is exactly the hand-built design matrix shown above followed by `summary`. Choose `polyfit` when the input is a single variable and the output benefits from being a `Polynomial` — evaluable, differentiable, composable. The error contracts also differ: `polyfit` returns `nil` on bad input, while `LinearRegression.fit` throws `MatrixError.singular`. See <doc:Linear-Regression> for the full inference treatment.

> Note: The internal design matrix becomes severely ill-conditioned as degree rises — losing roughly 9 digits of double precision at degree 4 and reaching the edge of representable precision around degree 5 on a typical input range. `polyfit` returns `nil` when the conditioning fails outright; for degrees above 4 on real data, prefer regularization or a basis transformation. See <doc:Numerical-Literacy>.

> Note: For the conceptual background on least squares (projection onto a column space) see <doc:Vector-Projections>. Polynomial regression projects `y` onto the column space spanned by `[1, x, x², ..., xⁿ]`.

### Coefficient ordering and trimming

Quiver stores coefficients in ascending order of power because that is the order arithmetic operations produce naturally and the order numerical solvers consume. The `asExpression` rendering uses descending order, with the highest-degree term first, because that matches how humans write polynomials. See <doc:Rendering-Math-Primer> for the full rendering family and the `relativeZeroTolerance` parameter that suppresses numerical noise in fitted polynomials. Both views describe the same value:

```swift
import Quiver

let p = Polynomial([1, 3, 2])
p.coefficients              // [1.0, 3.0, 2.0]   — ascending: a₀, a₁, a₂
p.asExpression()             // "2x² + 3x + 1"     — descending, human-readable
```

Two polynomial values are `Equatable` when their coefficient arrays match exactly. Operations like `+` and `*` may introduce trailing zeros. A polynomial of degree two added to its negative is mathematically zero, but the resulting array may still carry trailing zeros that survive the addition. The `trimmed` property returns the canonical form by stripping trailing zeros, which is the right call before comparing two polynomials for equality:

```swift
import Quiver

let a = Polynomial([1, 2, 0, 0])
let b = Polynomial([1, 2])

a == b                       // false — coefficient arrays differ in length
a.trimmed() == b             // true  — canonical forms match
```

The `degree` property reports the highest power with a non-zero coefficient regardless of trailing zeros, so `Polynomial([1, 2, 0])` reports a degree of `1`, not `2`. The zero polynomial, `Polynomial([0])`, has degree `0` by convention, the same as any other constant.

> Experiment: **The Quiver Notebook** is the right surface for watching polynomial fits go wrong. Sweep the degree from 1 to 8 and re-evaluate — R² on the training rows keeps rising, but the curve starts to chase noise between points. The gap between training fit and out-of-sample behaviour is why holdout matters. See <doc:Quiver-Notebook>.

## Topics

### Creating a polynomial
- ``Polynomial/init(_:)``

### Properties
- ``Polynomial/coefficients``
- ``Polynomial/degree``

### Evaluation
- ``Polynomial/callAsFunction(_:)-(Double)``
- ``Polynomial/callAsFunction(_:)-([Double])``

### Calculus
- ``Polynomial/derivative()``

### Rendering
- ``Polynomial/asExpression(relativeZeroTolerance:)``

### Canonical form
- ``Polynomial/trimmed()``

### Fitting
- ``Swift/Array/polyfit(x:y:degree:)``

### Related
- <doc:Vector-Projections>

