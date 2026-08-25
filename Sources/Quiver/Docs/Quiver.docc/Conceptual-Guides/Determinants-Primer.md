# Linear Independence and Determinants

Understanding when a set of vectors carries genuinely new information.

## Overview

A set of vectors is **linearly independent** when each vector points in a direction that cannot be reached through a linear combination (a scaled sum) of the others. Each independent vector contributes a new dimension to the space. Conversely, a vector is redundant if it can be expressed as a combination of other vectors in the set. In this article, we'll review linear independence and how to measure a vector space using determinants.

> Note: This primer builds on the vectors of the <doc:Linear-Algebra-Primer>, the matrices of <doc:Shape-And-Size> and <doc:Matrix-Operations>, and the transformations of <doc:Matrix-Transformations>.

### Span and independence

Consider a single vector b₁. Scaling it — stretching it, shrinking it, flipping it negative — traces out one line through the origin:

```swift
import Quiver

let b1 = [1.0, 3.0]
// Every multiple of b1 lies on the same line:
// [2.0, 6.0], [0.5, 1.5], [-1.0, -3.0] ...
```

The set of points a collection of vectors can reach through linear combinations is its **span**. The span of one non-zero vector is a single line. One vector alone cannot be dependent — there is nothing yet for it to depend on.

#### A second direction, or the same one again

Add a second vector and ask one question: does it supply a genuinely different direction, or does it merely retrace b₁'s line? For two vectors, **dependent** means one is a scalar multiple of the other. Parallel vectors have a `cosineOfAngle` of exactly `1.0` (or `-1.0` when they point opposite ways):

```swift
let b1 = [1.0, 3.0]

let b2 = [2.0, 1.0]
b1.cosineOfAngle(with: b2)   // 0.7071067811865475 — not ±1, a genuinely new direction

let b3 = 2.0 * b1            // [2.0, 6.0]
b1.cosineOfAngle(with: b3)   // 0.9999999999999998 — the same line wearing a longer arrow
```

The pair b₁ and b₂ is linearly independent. Neither is a multiple of the other, so each contributes a direction the other cannot reach. Their span is no longer a line but the entire plane. The pair b₁ and b₃ is dependent. Because b₃ was manufactured as `2.0 * b1`, its direction is already spoken for, and the span of the pair is still just b₁'s original line.

#### Basis and dimension

A **basis** is a set of linearly independent vectors that span an entire space. The number of vectors in that basis defines the space's **dimension**. The pair b₁ and b₂ forms a basis for the 2D plane. The dependent pair b₁ and b₃ is not a basis at all—it merely offers two names for a single direction.

Basis vectors do not need to be unit length or sit at right angles to each other. However, an orthonormal basis (mutually perpendicular unit vectors, like the standard `[1, 0]` and `[0, 1]`) is the most convenient, as it makes dot products and cosine calculations behave perfectly.

> Note: In statistics, **independent** means two variables share no information. Linear independence is different: it just means vectors cannot be built from one another. Two variables can be highly correlated statistically and yet, as vectors, still be linearly independent.

### A system solved in one call

Deciding independence for anything larger than a pair takes an instrument that can search for combinations. Quiver's is `solve(_:)`.

A linear equation is one whose unknowns appear only to the first power — geometrically, a flat constraint like a line or a plane. A system of linear equations asks for the values that satisfy several of these constraints at once, packaged as `Ax = b`. Consider a café where two receipts survive from the morning, and we want the price of a coffee and the price of a muffin:

```
Receipt 1:  2 coffees + 1 muffin  = $5
Receipt 2:  1 coffee  + 3 muffins = $10
```

The item counts form the matrix `A` and the totals form the vector `b`. The `solve(_:)` method takes the right-hand side and returns the unknowns `x` that satisfy both rows at once:

```swift
// 2c + 1m = 5, 1c + 3m = 10
let A = [[2.0, 1.0],
         [1.0, 3.0]]
let b = [5.0, 10.0]

A.solve(b)   // Optional([1.0, 3.0]) — coffee $1, muffin $3
```

#### When no unique answer exists

Two lines cross at one point only when they point in different directions. Suppose the café's second receipt is lost, and we replace it with a duplicate order — exactly the first receipt, doubled:

```swift
// Receipt 2 is now 2 × Receipt 1 — linearly dependent
let dependent = [[2.0, 1.0],
                 [4.0, 2.0]]

// Redundant: doubled order, doubled total (same line twice)
dependent.solve([5.0, 10.0])  // nil — infinitely many price pairs fit

// Contradictory: doubled order, incompatible total (parallel lines)
dependent.solve([5.0, 11.0])  // nil — no price pair fits
```

The matrix is **singular** — it has no inverse and no single answer to return, because one row is a scalar multiple of the other. The two equations either describe the same line twice (infinitely many solutions), or they describe parallel lines that never meet (no solution). `solve(_:)` returns `nil` in both cases.

Under the hood, `solve(_:)` computes `x = A⁻¹b`, so the `nil` is an inversion that could not happen. The expanded form, `try b.transformedBy(A.inverted())`, throws `MatrixError.singular` instead.

#### The combination test

To find a linear combination that satisfies a target, we hand the search to `solve(_:)`. Place b₁ and b₂ into a matrix as columns, give it a target, and the coefficients it returns are the combination:

```swift
// Which combination of b1 and b2 reaches [4, 7]?
// a1 · [1, 3] + a2 · [2, 1] = [4, 7]
let columns = [[1.0, 2.0],
               [3.0, 1.0]]     // each inner array is a row; b1 and b2 run down the columns
columns.solve([4.0, 7.0])      // Optional([2.0, 1.0]) — a1 = 2, a2 = 1
```

Two b₁'s plus one b₂ lands exactly on `[4, 7]`. Because b₁ and b₂ are independent in a two-dimensional space, they span the whole plane and this search succeeds for *every* target. Any third vector added to this plane is automatically dependent. Run the same search on a dependent pair and it fails the way the duplicated receipt did — both columns point down one line, and no combination of copies of a single direction can ever leave that line.

### The determinant tells them apart

The two café matrices looked almost alike, yet one returned prices and the other returned `nil`. Every square matrix carries a single fundamental number that separates the two cases. The **determinant** is non-zero exactly when the matrix's columns are independent, and zero the moment any column is a combination of the others:

```swift
let A = [[2.0, 1.0],
         [1.0, 3.0]]
A.determinant          // (2 × 3) − (1 × 1); independent columns

let dependent = [[2.0, 1.0],
                 [4.0, 2.0]]
dependent.determinant  // (2 × 2) − (1 × 4); dependent columns
```

For a `2`×`2` matrix the determinant is `ad - bc` — multiply down each diagonal and subtract. The number came back non-zero for the solvable system and exactly zero for the singular one.

### Scaling space

The determinant's home ground is geometry. A matrix transforms every vector in a coordinate system, and the determinant measures how much the transformation scales area. The simplest case is a diagonal matrix — one that scales each axis on its own, leaving the axes at right angles. Consider a matrix that stretches space by a factor of `3` horizontally and `5` vertically:

```swift
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

The `ad - bc` shortcut from the café matrices is the signed area of the parallelogram the transformed basis vectors draw:

```swift
let general = [[3.0, 1.0],
               [2.0, 5.0]]
general.determinant  // 13.0 = (3 × 5) - (1 × 2)
```

The sign matters. A negative determinant means the transformation flips orientation, like looking at space in a mirror. A rotation is the opposite case — it preserves both area and orientation, so its determinant is always `1`.

#### Higher dimensions

The same idea carries into higher dimensions. For a `3`×`3` matrix, the determinant measures volume scaling. For an `n`×`n` matrix, it measures the scaling of n-dimensional volume. Quiver handles any size:

```swift
let matrix3x3 = [
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 4.0],
    [5.0, 6.0, 0.0]
]
matrix3x3.determinant  // 0.9999999999999964

```

### When the determinant is zero

A determinant of zero means the matrix is singular. In other words its columns are linearly dependent, and the transformation crushes space into a lower dimension. A matrix's columns determine where the basis vectors land. Land them in different directions, and they fence off a parallelogram with real area. Land them on the same line, and the parallelogram is flat: the plane is crushed onto a line, and the map cannot be undone.

Consider a matrix where both columns point in the same direction:

```swift
let singular = [[1.0, 2.0],
                [1.0, 2.0]]
singular.determinant  // 0.0
```

Both transformed basis vectors land on the line `y = x`. Every point in 2D space now maps onto that single line — an entire dimension of information is gone. We cannot rebuild the original 2D positions from a 1D line, so no inverse exists.

![Determinant Singular](diagram-determinant-singular)

The same thing happens in larger matrices: when one row is a combination of the others, that row carries no new information, the determinant is zero, and the system has no unique solution — three unknowns, say, but only two independent equations.

#### Redundancy is dependence in data

The duplicated receipt was linear dependence in data: the doubled order was a scalar multiple of the first, offering no new information. In machine learning, a feature matrix may carry a column that is a linear combination of the others. Those features are **collinear** — linear dependence read across the columns of a data table rather than the rows of an equation system. Temperature recorded in both Celsius and Fahrenheit is the textbook case: the second column carries no information the first did not.

Singular matrices appear more often than we might expect. In computer graphics, a transformation that projects 3D objects onto a 2D screen is singular by nature — we lose the depth dimension, and that loss is the purpose.

### Matrix inversion

A non-zero determinant means a matrix can be inverted to undo its transformation. Multiplying a matrix by its inverse always yields the identity matrix. 

Because the inverse undoes the original scaling, its determinant is the exact reciprocal. If the original transformation scaled the area by `13`, the inverse must scale it by `1/13` to restore the original size:

```swift
let general = [[3.0, 1.0],
               [2.0, 5.0]]

let inv = try general.inverted()

// Yields the identity matrix (with slight floating-point noise)
general.multiplyMatrix(inv)  // ≈ [[1.0, 0.0], [0.0, 1.0]]

general.determinant          // 13.0
inv.determinant              // 0.0769... (1/13)
```

> Important: Calling `.inverted()` on a singular matrix throws a `MatrixError/singular` error.

```swift
let matrix = [[1.0, 2.0],
              [2.0, 4.0]]
do {
    let inverse = try matrix.inverted()
} catch {
    print("Matrix is singular — no inverse exists")
}
```

#### Fractional display

The decimal result `0.0769...` hides the relationship underneath. The denominator is `13` because the determinant is `13`. The `asFractions` method brings that structure back into view:

```swift
let inverse = try general.inverted()
inverse.asFractions()
// [[5/13, -1/13],
//  [-2/13, 3/13]]
```

> Tip: Every element shares the determinant as its denominator. The ``Fraction`` type is presentation-only — every operation still runs on standard `Double` values internally. See <doc:Rendering-Math-Primer> for the full rendering chain.

### Near dependence and the condition number

Exact dependence is rare in measured data. Near dependence is common. Slide b₂ from `[2.0, 1.0]` toward a near-multiple of b₁ and re-run the combination search for `[4.0, 7.0]`:

```swift
// b2 now almost parallel to b1 = [1, 3]
let nearlyOneLine = [[1.0, 2.0],
                     [3.0, 5.9]]
nearlyOneLine.solve([4.0, 7.0])   // Optional([-96.00000000000034, 50.00000000000017])
```

The search succeeds, but the answer relies on ninety-six negative copies of one vector fighting fifty copies of the other. The target is reachable, but only by having large, opposite contributions nearly cancel. The answer is so sensitive that a tiny change in the data rewrites it completely.

The **condition number** measures this sensitivity. Exact zeros are rare in floating-point math — the interesting cases sit just short of collapse — and the condition number reports how close a matrix sits to that edge: how much a small change in the input can swing the result. A well-conditioned matrix amplifies small input changes only slightly. An ill-conditioned matrix amplifies them enormously:

```swift
let nearSingular = [[1.0, 1.0],
                    [1.0, 1.0000001]]
nearSingular.determinant      // 1.0000000005838672e-07 (≈ 0.0000001 — non-zero, technically invertible)
nearSingular.conditionNumber  // > 1,000,000 (inversion results are unreliable)
```

This matrix has a tiny but non-zero determinant. Calling `try inverted()` returns a result, and that result is numerically unreliable. The condition number catches what the determinant alone misses — it is the continuous version of the independence test, measuring *how nearly* dependent the columns are rather than answering yes or no.

**Interpreting the condition number:**

These thresholds describe the condition number of the matrix we invert directly. A regression inverts `XᵀX` instead — its condition number is roughly the square of the data's own, so the trustworthy bar there is stricter. See <doc:Model-Interpretation-Primer>.

- Near `1.0`: Well-conditioned, safe to invert
- `10³`–`10⁶`: Moderate conditioning, results may lose precision
- Above `10⁶`: Ill-conditioned, inversion results are unreliable
- Infinity: Singular matrix, no inverse exists

### The failure a model meets

Collinear features make the matrix a model must invert singular. `LinearRegression` fits by solving the normal equation θ = (XᵀX)⁻¹Xᵀy, and that formula inverts `XᵀX`. When the features are linearly dependent, the `XᵀX` it inverts is singular, and `fit(features:targets:)` throws `MatrixError.singular`. A zero determinant is the mathematical signature of this collapse; the solver detects it when the elimination it runs turns up a vanishingly small pivot.

The near-dependent case is quieter and more common. `XᵀX` stays technically invertible, its condition number races upward, and the coefficients take on the same lopsided, opposite-signed character as the ninety-six-versus-fifty combination above. The standard absorber is ``Ridge``: its L2 penalty stabilizes a near-singular `XᵀX`, trading a little training-set accuracy for a fit that no longer collapses. The <doc:Model-Interpretation-Primer> reads the condition number before a fit and the coefficients after one; the <doc:Regularization-Primer> explains the penalty itself.

### One fact, several faces

Everything this primer measures is one property viewed from different sides. The opening sections defined that property with hand tools; the determinant is where it became a single number; and the consequences fan out from there.

> Important: For a square matrix `A`, the columns are linearly independent ⇔ the `determinant` is non-zero ⇔ `A` is **invertible** ⇔ `Ax = b` has exactly one solution for every `b` ⇔ the transformation **preserves dimension** ⇔ the `conditionNumber` is finite. Negate one statement and every other statement falls with it.

The count these faces share has a name. **Rank** is the dimension of the span — the count of independent directions a set of vectors actually delivers. Two independent columns have rank `2` and span the plane; make one column a multiple of the other and the rank drops to `1`, and the pair spans only a line. Quiver exposes no `rank` property, so we read rank deficiency off its symptoms: a zero determinant, and a `solve(_:)` that returns `nil`.

Where these faces earn their keep is model fitting. The <doc:Model-Interpretation-Primer> turns the condition number and the coefficients into a trust report for a fitted model, and the <doc:Regularization-Primer> shows how a penalty restores a solvable problem when the features are nearly dependent. One more face waits in the <doc:Eigenvalues-Primer>: the determinant is the product of the matrix's eigenvalues — the characteristic stretch factors that lead on to principal component analysis.

> Experiment: Use **The Quiver Notebook** (<doc:Quiver-Notebook>) to watch every diagnostic fail at once by intentionally sliding a matrix into singularity. Set `b1 = [1.0, 3.0]` alongside a target of `[4.0, 7.0]`, sweep the second column slowly toward exact dependence at `[2.0, 6.0]`, and print the metrics at each step. As the vectors become parallel, watch the cosine climb to `1.0`, the `solve(_:)` coefficients explode, the determinant drop to zero, and the condition number race toward infinity until the solver finally returns `nil`.
