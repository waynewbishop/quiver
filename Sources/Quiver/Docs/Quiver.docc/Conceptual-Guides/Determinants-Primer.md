# Determinants Primer

Understand how matrices scale space and assess numerical stability.

## Overview

Every square matrix has a single number associated with it called the **determinant**. This value answers a geometric question: when the matrix transforms space, how much does the area (or volume) change? The determinant tells us whether a transformation is reversible and how it scales the space it acts on.

> Note: This primer builds on concepts introduced in the <doc:Linear-Algebra-Primer>, <doc:Shape-And-Size>, and <doc:Matrix-Transformations>.

### Scaling space

A matrix transforms every vector in a coordinate system. The simplest case is a diagonal matrix, which scales each axis independently. Consider a matrix that stretches space by a factor of `3` horizontally and `5` vertically:

```swift
import Quiver

let scale = [[3.0, 0.0],
             [0.0, 5.0]]
scale.determinant  // 15.0
```

The original `1`×`1` unit square becomes a `3`×`5` rectangle with area `15`. The determinant captures this scaling factor directly, telling us that every region in the original space is now `15` times larger.

Adding a shear component tilts the rectangle into a parallelogram, but the area does not change. The base-times-height calculation still produces the same result:

```swift
let shear = [[3.0, 0.0],
             [2.0, 5.0]]
shear.determinant  // 15.0
```

For a general `2`×`2` matrix, the determinant equals `ad - bc`. This formula computes the signed area of the parallelogram formed by the transformed basis vectors:

```swift
let general = [[3.0, 1.0],
               [2.0, 5.0]]
general.determinant  // 13.0 = (3 × 5) - (1 × 2)
```

The sign matters. A negative determinant means the transformation flips orientation — like looking at space in a mirror. A rotation matrix, which preserves both area and orientation, always has a determinant of `1`:

```swift
// 90° counterclockwise rotation
let rotation = [[0.0, -1.0],
                [1.0,  0.0]]
rotation.determinant  // 1.0 (area preserved, no flip)
```

### Higher dimensions

The same principle extends naturally. For a `3`×`3` matrix, the determinant measures volume scaling. For an `n`×`n` matrix, it measures the scaling of n-dimensional volume. Quiver handles any size:

```swift
let matrix3x3 = [
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 4.0],
    [5.0, 6.0, 0.0]
]
matrix3x3.determinant  // 1.0
```

#### When the determinant is zero

A determinant of zero signals that the matrix is **singular**. It collapses space into a lower dimension. In 2D, the transformation flattens everything onto a line. In 3D, it compresses a volume into a plane or a line.

Consider a matrix where both columns point in the same direction:

```swift
let singular = [[1.0, 2.0],
                [1.0, 2.0]]
singular.determinant  // 0.0
```

Both transformed basis vectors land on the line `y = x`. Every point in 2D space gets mapped onto that single line, losing an entire dimension of information. We cannot reconstruct the original 2D positions from a 1D line, so no inverse exists.

The same principle applies to larger matrices. When one row is a linear combination of the others, the system of equations the matrix represents is underdetermined:

```swift
let dependent = [
    [1.0, 1.0, 3.0],
    [1.0, 2.0, 4.0],
    [2.0, 3.0, 7.0]   // row 1 + row 2
]
dependent.determinant  // 0.0
```

The third equation adds no new information. We have three unknowns but only two independent equations. That is not enough to find a unique solution.

> Important: Calling `.inverted()` on a singular matrix throws a `MatrixError.singular` error. Handle this with `do-catch`, `try?`, or check the determinant first:

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

Singular matrices appear more often than we might expect. In machine learning, a feature matrix becomes singular when one feature is a perfect linear combination of others. In computer graphics, a transformation that projects 3D objects onto a 2D screen is inherently singular — we lose the depth dimension, and that loss is by design.

### Matrix inversion

When a matrix has a non-zero determinant, we can compute its inverse. The inverse matrix undoes the original transformation: if matrix A rotates a vector `90`° clockwise, then A⁻¹ rotates it `90`° counterclockwise, returning it to its original position.

```swift
let A = [[3.0, 1.0],
         [2.0, 5.0]]

let inv = try A.inverted()
let identity = A.multiplyMatrix(inv)
// [[1.0, 0.0],
//  [0.0, 1.0]]
```

The product A × A⁻¹ always equals the identity matrix — the transformation that leaves everything unchanged. The determinant connects directly to inversion: the determinant of the inverse equals the reciprocal of the original determinant:

```swift
A.determinant                  // 13.0
try A.inverted().determinant   // 0.0769... (1/13)
```

This makes geometric sense. If the original transformation scales area by a factor of `13`, undoing that transformation must scale area by `1/13` to restore the original size.

#### Fractional display

The decimal result `0.0769...` obscures the underlying relationship — the denominator is `13` because the determinant is `13`. The `asFractions()` method reveals this structure:

```swift
let A = [[3.0, 1.0],
         [2.0, 5.0]]

let inverse = try A.inverted()
inverse.asFractions()
// [[5/13, -1/13],
//  [-2/13, 3/13]]

A.determinant.asFraction()  // 13
```

Every element shares the determinant as its denominator. This pattern is hidden by decimal representation. The `Fraction` type is presentation-only; all operations continue to use standard `Double` values internally. Use `asFractions()` on any `[Double]` or `[[Double]]` result, or `asFraction()` on a single `Double`.

### Solving linear systems

One of the most practical applications of matrix inversion is solving systems of linear equations. Given the system Ax = b, the solution is x = A⁻¹b:

```swift
// System: 3x + y = 10, 2x + 5y = 27
let A = [[3.0, 1.0],
         [2.0, 5.0]]
let b = [10.0, 27.0]

let solution = try b.transformedBy(A.inverted())
// [1.77, 4.69]
```

This only works when the determinant is non-zero. A zero determinant means the equations are either contradictory (no solution) or redundant (infinitely many solutions).

### Condition number

A matrix can have a non-zero determinant and still produce unreliable results when inverted. The **condition number** quantifies this risk by measuring how sensitive the result is to small changes in the input — think of it like a lever. A well-conditioned matrix amplifies small input changes only slightly, while an ill-conditioned matrix amplifies them enormously.

```swift
let identity = [[1.0, 0.0],
                [0.0, 1.0]]
identity.conditionNumber  // 1.0 (perfectly conditioned)
```

A condition number near `1.0` means the matrix is well-behaved. As the value grows, the matrix becomes increasingly ill-conditioned, and small rounding errors in the input get amplified into large errors in the output:

```swift
let nearSingular = [[1.0, 1.0],
                    [1.0, 1.0000001]]
nearSingular.determinant      // 0.0000001 (non-zero, technically invertible)
nearSingular.conditionNumber  // > 1,000,000 (inversion results are unreliable)
```

This matrix has a tiny but non-zero determinant, so `try inverted()` returns a result — but that result is numerically unreliable. The condition number catches what the determinant alone misses.

**Interpreting the condition number:**

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

In production code, checking the condition number before inverting a matrix prevents silent numerical failures. A recommendation engine computing user-item similarity matrices, a physics simulation solving force equations, or a calibration system fitting sensor data — all benefit from knowing whether their matrix is safe to invert before trusting the result.

```swift
let matrix = [[4.0, 1.0],
              [1.0, 3.0]]

let cond = matrix.conditionNumber
if cond < 1_000 {
    let inverse = try matrix.inverted()
    // Safe to use the inverse
} else {
    // Matrix is ill-conditioned — consider regularization
}
```

### How Quiver uses determinants

The determinant and matrix inversion are not just abstract concepts — they power linear regression behind the scenes. When we call `LinearRegression.fit(features:targets:)`, Quiver solves the normal equation θ = (X'X)⁻¹X'y to find the best-fit coefficients. That formula requires inverting the matrix `X'X`, which is only possible when its determinant is non-zero.

If the feature vectors are linearly dependent — for example, temperature in both Celsius and Fahrenheit — the matrix `X'X` becomes singular (`determinant = 0`) and the inversion fails. That is why `fit` throws `MatrixError.singular`. This is the determinant telling us that the features do not contain enough independent information to solve the problem.

> Tip: The same math used in <doc:Matrix-Transformations> to rotate and scale points is what that `LinearRegression` applies to find coefficients. The difference is context: in graphics we transform geometry, in regression we solve for the line that best fits the data.

### Putting it all together

The determinant and condition number form a diagnostic pair. Before inverting a matrix, we can verify the operation is safe:

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

For matrices that fail these checks, we know to handle the situation gracefully — whether that means removing redundant features, adjusting the data, or reporting that the computation cannot be performed reliably.

### See also

- <doc:Matrix-Operations> - Matrix arithmetic, transpose, and multiplication
- <doc:Matrix-Transformations> - Matrix-vector transformations and basis vectors
- <doc:Composing-Transformations> - Composing multiple transformations
- <doc:Linear-Algebra-Primer> - Linear algebra fundamentals
