# Determinants

Understand how matrices scale space and assess numerical stability.

## Overview

In linear algebra, a square matrix can be thought of as a transformation that stretches, squashes, or rotates space. Every square matrix carries a single fundamental number called the **determinant** that measures the magnitude of this change. This number answers one geometric question: when the matrix transforms space, it changes the overall area or volume by some factor — and the determinant is that exact factor. 

The same number also answers the algebraic question the previous primer left open. The determinant is the single-value test for linear independence, returning a non-zero value exactly when the matrix's columns are independent, and returning zero the moment any column is a combination of the others. The determinant tells us three things at once: how much a transformation scales the coordinate space, whether its columns carry genuinely independent directions, and whether the transformation can be undone.

> Note: This primer builds on the <doc:Linear-Independence-Primer>, which defines the property the determinant tests for and shows it first as a failure — a system with no unique answer. This primer also draws on concepts introduced in the <doc:Linear-Algebra-Primer>, <doc:Shape-And-Size>, and <doc:Matrix-Transformations>.

### Scaling space

A matrix transforms every vector in a coordinate system. The simplest case is a diagonal matrix — one that scales each axis on its own, leaving the axes at right angles. Consider a matrix that stretches space by a factor of `3` horizontally and `5` vertically:

```swift
import Quiver

let scale = [[3.0, 0.0],
             [0.0, 5.0]]
scale.determinant  // 15.0
```

The original `1`×`1` unit square becomes a `3`×`5` rectangle with area `15`. The determinant captures this scaling factor directly. Every region in the original space is now `15` times larger.

![Determinant Scaling](diagram-determinant-scaling)

Adding a shear component tilts the rectangle into a leaning parallelogram. The shape leans, but the area does not change. Base times height still gives the same result:

```swift
let shear = [[3.0, 0.0],
             [2.0, 5.0]]
shear.determinant  // 15.0
```

Every `2`×`2` matrix has a shortcut for this area. The determinant equals `ad - bc`. That formula is the signed area of the parallelogram the transformed basis vectors draw — the same parallelogram the <doc:Linear-Independence-Primer> watched flatten as its columns slid toward one line:

```swift
let general = [[3.0, 1.0],
               [2.0, 5.0]]
general.determinant  // 13.0 = (3 × 5) - (1 × 2)
```

The sign matters. A negative determinant means the transformation flips orientation, like looking at space in a mirror. A rotation is the opposite case — it preserves both area and orientation, so its determinant is always `1`:

```swift
// 90° counterclockwise rotation
let rotation = [[0.0, -1.0],
                [1.0,  0.0]]
rotation.determinant  // 1.0 (area preserved, no flip)
```

### Higher dimensions

The same idea carries into higher dimensions. For a `3`×`3` matrix, the determinant measures volume scaling. For an `n`×`n` matrix, it measures the scaling of n-dimensional volume. Quiver handles any size:

```swift
let matrix3x3 = [
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 4.0],
    [5.0, 6.0, 0.0]
]
matrix3x3.determinant  // 1.0
```

#### When the determinant is zero

A determinant of zero is the signal to watch for. It means the matrix is **singular** — its columns are linearly dependent, and the transformation crushes space into a lower dimension. This is the independence test failing inside a matrix: the flattened parallelogram of the <doc:Linear-Independence-Primer>, now reported as a zero. In 2D, the transformation flattens everything onto a line. In 3D, it presses a volume down into a plane or a line.

Consider a matrix where both columns point in the same direction:

```swift
let singular = [[1.0, 2.0],
                [1.0, 2.0]]
singular.determinant  // 0.0
```

Both transformed basis vectors land on the line `y = x`. Every point in 2D space now maps onto that single line — an entire dimension of information is gone. We cannot rebuild the original 2D positions from a 1D line, so no inverse exists.

![Determinant Singular](diagram-determinant-singular)

The same thing happens in larger matrices. When one row is a combination of the others, that row carries no new information. The system the matrix represents has no unique solution:

```swift
let dependent = [
    [1.0, 1.0, 3.0],
    [1.0, 2.0, 4.0],
    [2.0, 3.0, 7.0]   // row 1 + row 2
]
dependent.determinant  // 0.0
```

The third equation adds no new information. We have three unknowns but only two independent equations. That is not enough to find a unique solution.

> Important: Calling `.inverted()` on a singular matrix throws a ``MatrixError/singular`` error. Handle this with `do-catch`, `try?`, or check the condition number first:

```swift
import Quiver

// Safe inversion with error handling
let matrix = [[1.0, 2.0],
              [2.0, 4.0]]

do {
    let inverse = try matrix.inverted()
} catch {
    print("Matrix is singular — no inverse exists")
}

// Or use try? for optional result
let maybeInverse = try? matrix.inverted()  // nil for singular matrices
```

#### Why this matters in practice

Singular matrices appear more often than we might expect. In machine learning, a feature matrix becomes singular when one feature is a perfect linear combination of others. In computer graphics, a transformation that projects 3D objects onto a 2D screen is singular by nature. We lose the depth dimension, and that loss is the whole point.

### Matrix inversion

When a matrix has a non-zero determinant, we can compute its inverse. The inverse undoes the original transformation. If matrix A rotates a vector `90`° clockwise, then A⁻¹ rotates it `90`° counterclockwise — and the vector returns to where it started.

```swift
let A = [[3.0, 1.0],
         [2.0, 5.0]]

let inv = try A.inverted()
let identity = A.multiplyMatrix(inv)
// [[1.0, 0.0],
//  [0.0, 1.0]]
```

The product A × A⁻¹ always equals the identity matrix, the transformation that leaves everything unchanged. The determinant connects directly to inversion. The determinant of the inverse is the reciprocal of the original determinant:

```swift
A.determinant                  // 13.0
try A.inverted().determinant   // 0.0769... (1/13)
```

This makes geometric sense. If the original transformation scales area by a factor of `13`, undoing that transformation must scale area by `1/13` to restore the original size.

#### Fractional display

The decimal result `0.0769...` hides the relationship underneath. The denominator is `13` because the determinant is `13`. The `asFractions` method brings that structure back into view:

```swift
let A = [[3.0, 1.0],
         [2.0, 5.0]]

let inverse = try A.inverted()
inverse.asFractions()
// [[5/13, -1/13],
//  [-2/13, 3/13]]

A.determinant.asFraction()  // 13
```

Every element shares the determinant as its denominator. Decimal representation hides that pattern. The ``Fraction`` type is presentation-only. Every operation still runs on standard `Double` values internally. Use `asFractions` on any `[Double]` or `[[Double]]` result, or `asFraction` on a single `Double`.

Matrix inversion also answers systems of equations. Solving `Ax = b` is applying `A⁻¹` to `b`. The `solve(_:)` method handles this, and the <doc:Linear-Independence-Primer> is where we first met the shapes that have no unique answer. This primer stays with what the determinant and condition number reveal about the matrix itself.

### Condition number

A matrix can have a non-zero determinant and still produce unreliable results when inverted. Exact zeros are rare in floating-point math — the interesting cases sit just short of collapse. The **condition number** measures how close a matrix sits to that edge. It reports how much a small change in the input can swing the result. A well-conditioned matrix amplifies small input changes only slightly. An ill-conditioned matrix amplifies them enormously.

```swift
let identity = [[1.0, 0.0],
                [0.0, 1.0]]
identity.conditionNumber  // 1.0 (perfectly conditioned)
```

A condition number near `1.0` means the matrix is well-behaved. As the value grows, the matrix drifts toward collapse. Small rounding errors in the input then grow into large errors in the output:

```swift
let nearSingular = [[1.0, 1.0],
                    [1.0, 1.0000001]]
nearSingular.determinant      // 0.0000001 (non-zero, technically invertible)
nearSingular.conditionNumber  // > 1,000,000 (inversion results are unreliable)
```

This matrix has a tiny but non-zero determinant. Calling `try inverted()` returns a result, and that result is numerically unreliable. The condition number catches what the determinant alone misses — it is the continuous version of the independence test, measuring *how nearly* dependent the columns are rather than answering yes or no.

**Interpreting the condition number:**

These thresholds describe the condition number of the matrix we invert directly. A regression inverts `XᵀX` instead. Its condition number is roughly the square of the data's own, so the trustworthy bar there is stricter. See <doc:Model-Interpretation-Primer>.

- Near `1.0`: Well-conditioned, safe to invert
- `10³`–`10⁶`: Moderate conditioning, results may lose precision
- Above `10⁶`: Ill-conditioned, inversion results are unreliable
- Infinity: Singular matrix, no inverse exists

```swift
let singular = [[1.0, 2.0],
                [2.0, 4.0]]
singular.conditionNumber  // .infinity
```

Quiver computes the condition number by comparing the largest absolute column sum of the matrix with that of its inverse.

#### When to check the condition number

In production code, checking the condition number before inverting a matrix prevents silent numerical failures. A recommendation engine builds user-item similarity matrices. A physics simulation solves force equations. A calibration system fits sensor data. Each one benefits from knowing whether its matrix is safe to invert before trusting the result.

```swift
let matrix = [[4.0, 1.0],
              [1.0, 3.0]]

let cond = matrix.conditionNumber
if cond < 1_000 {
    let inverse = try matrix.inverted()
    // Safe to use the inverse
} else {
    // Matrix is ill-conditioned — penalize the weights with Ridge or drop the redundant column
}
```

Collinearity zeroes this determinant when it appears in a real feature matrix. The fix is not more arithmetic but a change of model or of features. The <doc:Model-Interpretation-Primer> diagnoses the signature, and the <doc:Regularization-Primer> supplies the penalty. The detection tools live there too, including the `conditionNumber` of `XᵀX` — a value roughly the square of the data's own.

### How Quiver uses determinants

The determinant and matrix inversion power linear regression behind the scenes. When we call `LinearRegression.fit(features:targets:)`, Quiver solves the normal equation θ = (X'X)⁻¹X'y to find the best-fit coefficients. That formula inverts the matrix `X'X`. An inverse exists only when `X'X` is non-singular. A *usable* inverse needs more than that — the matrix must also be well-conditioned.

Suppose two feature vectors are linearly dependent — temperature in both Celsius and Fahrenheit, say. The matrix `X'X` becomes singular, the inversion has no solution, and `fit` throws `MatrixError.singular`. (A zero determinant is the mathematical signature of this collapse; the solver detects it when the elimination it runs turns up a vanishingly small pivot.) This is the determinant telling us that the features do not carry enough independent information to solve the problem. The <doc:Model-Interpretation-Primer> uses this same pair as an advance warning. It reads the condition number before a fit and the lopsided coefficients after one, to judge whether a model can be trusted.

> Note: The same math used in <doc:Matrix-Transformations> to rotate and scale points is what ``LinearRegression`` applies to find coefficients. Only the context differs. In graphics we transform geometry. In regression we solve for the line that best fits the data.

### One fact, several faces

Everything this primer measures is one property viewed from different sides. The <doc:Linear-Independence-Primer> defines that property with hand tools; the determinant is where it becomes a single number; and the consequences fan out from there.

> Important: For a square matrix `A`, the columns are linearly independent ⇔ the `determinant` is non-zero ⇔ `A` is **invertible** ⇔ `Ax = b` has exactly one solution for every `b` ⇔ the transformation **preserves dimension** ⇔ the `conditionNumber` is finite. Negate one statement and every other statement falls with it.

Independence is the root property. The determinant is the test that reports it as a single value. Invertibility is the consequence for the matrix, and the one-answer-or-none split is the consequence for systems. The `conditionNumber` is the continuous version of the same test, catching near-dependence where an exact zero would see nothing. One more face waits in the <doc:Eigenvalues-Primer>: the determinant is the product of the matrix's eigenvalues, the characteristic stretch factors that the primer builds from geometry.

### Putting it all together

The determinant and condition number form a diagnostic pair. Before inverting a matrix, we can check that the operation is safe:

```swift
let matrix = [[4.0, 7.0],
              [2.0, 6.0]]

// Is it singular?
let det = matrix.determinant  // 10.0 — non-zero, good

// Is it numerically stable?
let cond = matrix.conditionNumber  // small value — safe to invert

// Both checks pass — safe to proceed
let inverse = try matrix.inverted()
```

For matrices that fail these checks, we handle the situation gracefully. We remove the redundant features, adjust the data, or report that the computation cannot be performed reliably.

> Experiment: **The Quiver Notebook** is the right place to feel how the determinant and condition number move together. Build a 2×2 matrix that starts well-conditioned, then sweep one element toward making the columns parallel — print `determinant` and `conditionNumber` at each step. Watching the determinant slide toward zero while the condition number races toward infinity is the fastest way to see why both diagnostics matter. See <doc:Quiver-Notebook>.
