# Polynomials

Represent, evaluate, differentiate, and fit single-variable polynomials.

## Overview

A **polynomial** is an expression of the form `a₀ + a₁x + a₂x² + ... + aₙxⁿ`, a linear combination of powers of a single variable. Polynomials describe trajectories, model curved trends in data, approximate complicated functions on a small interval, and underpin a great deal of numerical computing. Quiver's `Polynomial` type is a value type that stores those coefficients and provides evaluation, arithmetic, differentiation, least-squares fitting, and algebraic rendering in one place.

The coefficients are stored in **ascending order of power**: `coefficients[0]` is the constant term, `coefficients[1]` is the linear term, `coefficients[2]` is the quadratic term, and so on. This matches the convention used in numerical computing libraries that work with polynomials as flat coefficient arrays. The `asExpression` method reverses the order to match human convention, rendering the highest-degree term first.

```swift
import Quiver

// 2x² + 3x + 1 — coefficients ascending: a₀=1, a₁=3, a₂=2
let p = Polynomial([1, 3, 2])

p(2)                    // 15.0  — evaluate at a single point
p([0, 1, 2, 3])         // [1.0, 6.0, 15.0, 28.0] — vectorized
p.derivative()          // 4x + 3
p.asExpression()        // "2x² + 3x + 1"
```

### Evaluating polynomials

The `Polynomial` type adopts Swift's `callAsFunction` so the value behaves like a Swift function. Calling `p(x)` evaluates the polynomial at a single `Double`; calling `p(xs)` on a `[Double]` evaluates at every point in the array, which is the right shape for plotting. Internally the evaluation uses **Horner's method**, rewriting `a₀ + a₁x + a₂x² + ... + aₙxⁿ` as `a₀ + x·(a₁ + x·(a₂ + ... + x·aₙ))` to avoid repeated `pow(x, k)` calls and the precision loss that comes with them:

```swift
import Quiver

// 2x² + 3x + 1
let p = Polynomial([1, 3, 2])

p(5)                    // 66.0
p(-1)                   // 0.0
p(0)                    // 1.0  — value of the constant term

// Vectorized evaluation across a grid of points — useful for charting
let xs = Array.linspace(start: -2.0, end: 2.0, count: 5)
p(xs)                   // [3.0, 0.0, 1.0, 6.0, 15.0]
```

Worth knowing what Horner's method is, even though we never write it ourselves. The textbook formula for evaluating `2 + 3x + 4x² + 5x³` walks each term, raises `x` to a power, multiplies by the coefficient, and sums — every power is its own rounding step, and the higher powers carry the largest errors. Horner's rewrite is the same expression nested: `2 + x·(3 + x·(4 + x·5))`. Each coefficient contributes one rounding step instead of one per power, and the work is `n` multiply-then-add operations instead of `n` calls to `pow`.

Quiver's `Polynomial` type uses Horner's method behind `callAsFunction`. Calling `p(x)` returns the precision-stable answer with no math required on the caller's side — no manual loop, no decision about evaluation order, no `pow` calls to manage. The teaching point is that the textbook formula and the stable formulation produce different floating-point results, and the library hides the choice so the user does not have to make it.

> Tip: <doc:Numerical-Literacy> covers the broader pattern of reformulating textbook formulas to keep floating-point error small. Horner's method is one example among several.

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

The derivative is a polynomial in its own right, so it composes with everything else: call `derivative` again for the second derivative, evaluate it at a grid of points, add it to another polynomial, or take its own derivative.

### Fitting polynomials to data

The `[Double].polyfit(x:y:degree:)` method is the least-squares counterpart to `LinearRegression.fit`. Given parallel `x` and `y` arrays and a target `degree`, it returns the polynomial of that degree that best fits the data in the ordinary-least-squares sense, the polynomial whose values minimize the sum of squared residuals against `y`. Internally `polyfit` builds a Vandermonde-style design matrix whose row `i` contains `[x[i], x[i]², ..., x[i]ⁿ]`, then defers to `LinearRegression.fit` to solve the normal equation:

```swift
import Quiver

// Underlying truth: 2x² + 3x + 1, evaluated at x = 1...5
let x = [1.0, 2.0, 3.0, 4.0, 5.0]
let y = [6.0, 15.0, 28.0, 45.0, 66.0]

if let p = [Double].polyfit(x: x, y: y, degree: 2) {
    p.coefficients   // ≈ [1.0, 3.0, 2.0]  — recovers the original polynomial
    p(6)             // ≈ 91.0  — predicted value at a new x
}
```

> Experiment: **The Quiver Notebook** is the right surface for watching polynomial fits go wrong. Sweep the degree from 1 to 8 and re-evaluate — R² on the training rows keeps rising, but the curve starts to chase noise between points. The gap between training fit and out-of-sample behaviour is why holdout matters. See <doc:Quiver-Notebook>.

Because `polyfit` is built on <doc:Linear-Regression>, calling `polyfit(degree: 1)` returns the same line that `LinearRegression.fit(features: x, targets: y)` would. Two doors to the same math. Higher degrees fit curves that linear regression cannot. The function returns `nil` when the inputs are invalid (mismatched lengths, fewer points than `degree + 1`, negative degree) or when the underlying linear system is ill-conditioned.

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
- <doc:Linear-Regression>
- <doc:Vector-Projections>

