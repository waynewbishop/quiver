# Linear Independence

Understand when a set of vectors carries genuinely new information, and why a system of equations can lack a unique answer.

## Overview

In Quiver, a vector is just a `[Double]` and a matrix is a `[[Double]]` whose columns are vectors. A column is **linearly independent** if it points somewhere the other columns cannot reach through a **linear combination** — a scaled sum of the others. If it can be reached, the column is redundant. The matrix carries the same span with one fewer vector.

This property carries practical stakes. A matrix of dependent columns is a classic way a model fit fails outright. In Quiver, it is the exact reason why a model like ``LinearRegression`` throws an error instead of returning a fit. 

### One vector reaches a line

Consider a single vector b₁. Scaling it — stretching it, shrinking it, flipping it negative — traces out one line through the origin, and that line is everything b₁ alone can reach:

```swift
import Quiver

let b1 = [1.0, 3.0]
// Every multiple of b1 lies on the same line:
// [2.0, 6.0], [0.5, 1.5], [-1.0, -3.0] ...
```

The set of points a collection of vectors can reach through linear combinations is its **span**. The span of one non-zero vector is a single line. One vector alone cannot be dependent — there is nothing yet for it to depend on.

### A second direction, or the same one again

Add a second vector and ask one question: does it supply a genuinely different direction, or does it merely retrace b₁'s line? For two vectors, "dependent" means one is a scalar multiple of the other. Parallel vectors have a `cosineOfAngle` of exactly `1.0` (or `-1.0` when they point opposite ways):

```swift
let b2 = [2.0, 1.0]
b1.cosineOfAngle(with: b2)   // 0.707 — not ±1, a genuinely new direction

let b3 = 2.0 * b1            // [2.0, 6.0]
b1.cosineOfAngle(with: b3)   // 0.9999999999999998 — the same line wearing a longer arrow
```

The pair b₁ and b₂ is **linearly independent**. Neither is a multiple of the other, so each contributes a direction the other cannot reach. Their span is no longer a line but the entire plane. The pair b₁ and b₃ is **dependent**. Because b₃ was manufactured as `2.0 * b1`, its direction is already spoken for, and the span of the pair is still just b₁'s original line. 

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

A well-posed system returns its answer wrapped in `Optional`. A system with no unique answer returns `nil`.

### When no unique answer exists

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

### The combination test

To find a linear combination that satisfies a target, we hand the search to `solve(_:)`. Place b₁ and b₂ into a matrix as columns, give it a target, and the coefficients it returns are the combination:

```swift
// Which combination of b1 and b2 reaches [4, 7]?
// a1 · [1, 3] + a2 · [2, 1] = [4, 7]
let columns = [[1.0, 2.0],
               [3.0, 1.0]]     // each inner array is a row; b1 and b2 run down the columns
columns.solve([4.0, 7.0])      // Optional([2.0, 1.0]) — a1 = 2, a2 = 1
```

Two b₁'s plus one b₂ lands exactly on `[4, 7]`. Because b₁ and b₂ are independent in a two-dimensional space, they span the whole plane and this search succeeds for *every* target. Any third vector added to this plane is automatically dependent.

Now run the same search on the dependent pair. Both columns point down one line, and no combination of copies of a single direction can ever leave that line:

```swift
// b1 and b3 share a line — can any combination reach [4, 7]?
let sameLine = [[1.0, 2.0],
                [3.0, 6.0]]    // b1 and b3 as columns
sameLine.solve([4.0, 7.0])     // nil — [4, 7] is off the line; no combination reaches it
```

### Basis and dimension

A **basis** is a set of n linearly independent vectors whose span is the whole space, and the space is then **n-dimensional**. Dimension is simply the count of independent directions a basis holds. 

Basis vectors do not have to be unit length or sit at right angles to each other. However, an **orthonormal** basis — mutually perpendicular unit vectors, like the natural basis `[1, 0]` and `[0, 1]` — is the one where dot products and cosines behave most simply.

### Redundancy is dependence in data

The duplicated receipt was linear dependence in data: the doubled order was a scalar multiple of the first, offering no new information. 

In machine learning, a feature matrix may carry a column that is a linear combination of the others. Those features are **collinear** — linear dependence read across the columns of a data table rather than the rows of an equation system. Temperature recorded in both Celsius and Fahrenheit is the textbook case. The second column carries no information the first did not.

### Near dependence, and a warning shape

Exact dependence is rare in measured data. Near dependence is common. Slide b₂ from `[2.0, 1.0]` toward a near-multiple of b₁ and re-run the search for `[4.0, 7.0]`:

```swift
// b2 now almost parallel to b1 = [1, 3]
let nearlyOneLine = [[1.0, 2.0],
                     [3.0, 5.9]]
nearlyOneLine.solve([4.0, 7.0])   // Optional([-96.00000000000034, 50.00000000000017])
```

The search succeeds, but the answer relies on ninety-six negative copies of one vector fighting fifty copies of the other. The target is reachable, but only by having large, opposite contributions nearly cancel. The answer is so sensitive that a tiny change in the data rewrites it completely. This sensitivity is measured by the condition number, explored further in the <doc:Determinants-Primer>.

### The geometry of collapse

A matrix's columns determine where the basis vectors land. Land them in different directions, and they fence off a parallelogram with real area. Land them on the same line, and the parallelogram is flat. The plane is crushed onto a line, and the map cannot be undone. 

### Two meanings of independent

The word *independent* has an older home in statistics, where it means two quantities carry no information about each other. That is a different idea from linear independence, which is about vectors and combinations. Two quantities can move together and yet, as vectors, still point in genuinely different directions. Everywhere this primer says *independent*, it means the vector sense.

### A word for what is lost

**Rank** is the dimension of the span — the count of independent directions a set of vectors actually delivers. Two independent columns have rank `2` and span the plane. Make one column a multiple of the other and the rank drops to `1`; the pair spans only a line.

Quiver exposes no `rank` property, so we read rank deficiency off its symptoms: a zero determinant, and a `solve(_:)` that returns `nil`.

### The failure a practitioner meets

Collinear features make the matrix a model must invert singular. ``LinearRegression`` fits by solving the normal equation, and when the features are linearly dependent, the `XᵀX` it inverts has lost rank. `fit(features:targets:)` throws `MatrixError.singular`. 

The standard absorber is ``Ridge``: its L2 penalty stabilizes a near-singular `XᵀX`, trading a little training-set accuracy for a fit that no longer collapses. See the <doc:Model-Interpretation-Primer> to learn how to catch collinearity before and after a fit.

> Experiment: **The Quiver Notebook** is the right place to watch independence fail one step at a time. Fix `b1 = [1.0, 3.0]` and a target of `[4.0, 7.0]`, then sweep the second column from `[2.0, 1.0]` toward `[2.0, 6.0]` — try `[2.0, 4.0]`, `[2.0, 5.9]`, `[2.0, 5.99]`, `[2.0, 6.0]` — printing `cosineOfAngle(with:)` and the `solve(_:)` combination at each step. The cosine climbs toward `1.0`, the combination coefficients explode into opposing pairs, and at exact dependence the search returns `nil`. See <doc:Quiver-Notebook>.
