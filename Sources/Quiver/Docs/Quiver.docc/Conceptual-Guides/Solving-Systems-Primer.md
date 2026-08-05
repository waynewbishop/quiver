# Solving Systems

Solve a system of linear equations with a single call, and recognize the shapes that have no unique answer.

## Overview

A linear equation is an algebraic equation where each term is either a constant or a variable raised to the power of one (e.g., `2x + 3y = 8`). Geometrically, a linear equation represents a perfectly flat object — a straight line in two-dimensional space, or a flat plane in three-dimensional space. The variables cannot have exponents, square roots, or trigonometric functions, as these twist or bend the graph. If we alter one variable, the other changes at a perfectly steady, predictable rate. 

A system of linear equations combines several of these equations to ask for the values that satisfy all of these flat constraints at once. For example, two lines that cross meet at exactly one point. Finding that point means finding the pair of numbers that make both equations true together. Linear algebra packages this multi-variable question into a compact form and solves the entire system in one step.

That form is `Ax = b`. Here, `A` holds the coefficients of our equations, `b` holds the right-hand constants, and `x` is the vector of unknowns we want to find. Quiver exposes this mathematical process directly through `solve(_:)`. A system that would traditionally take several lines of manual hand-elimination becomes a single, high-performance method call.

> Note: This primer builds on the vector and matrix concepts in the <doc:Linear-Algebra-Primer>. It stands on its own, and two later primers take its ideas further. The <doc:Linear-Independence-Primer> explores the property behind a system with no unique answer, and the <doc:Determinants-Primer> follows it with the geometry of singularity and the condition number.

### A system as a matrix equation

Consider a café problem. Two receipts survive from the morning, and we want the price of a coffee and the price of a muffin:

```
Receipt 1:  2 coffees + 1 muffin  = $5
Receipt 2:  1 coffee  + 3 muffins = $10
```

Each receipt constrains the same pair of unknowns. The item counts form a matrix, and the totals form a vector. Reading down each column of the matrix recovers one item's counts. Reading across each row recovers one receipt:

```swift
import Quiver

// 2c + 1m = 5, 1c + 3m = 10
let A = [[2.0, 1.0],
         [1.0, 3.0]]
let b = [5.0, 10.0]
```

The matrix `A` and the vector `b` together are the system. What we are missing is `x`, the vector of unknowns — one coffee price and one muffin price — that satisfies both rows at once. In `Ax = b` terms, `A` and `b` are known. The vector `x` is what we solve for.

### Solving in one call

The `solve(_:)` method takes the right-hand side `b` and returns the vector `x` that satisfies the system. This is the method's native role: pinning down the unknown prices that make every receipt add up at once:

```swift
let x = A.solve(b)
// Optional([1.0, 3.0])
```

Coffee is `$1` and a muffin is `$3`. We can check by substituting back into the receipts. The first gives `2 · 1 + 1 · 3 = 5`. The second gives `1 · 1 + 3 · 3 = 10`. Both totals hold, so the pair is correct.

The return type is optional. A well-posed system with a unique answer returns that answer wrapped in `Optional`. A system with no unique answer returns `nil`. We reach the `nil` case in a later section.

### The inverse expansion

The single call hides a two-step calculation. Seeing the steps makes the geometry visible. Solving `Ax = b` for `x` means undoing whatever `A` does to `x`. Undoing a transformation is exactly what the inverse `A⁻¹` is for. Multiply both sides by `A⁻¹` and the left side collapses to `x`. That leaves `x = A⁻¹b`.

Quiver lets us write that expanded form directly. Inverting `A` and applying the result to `b` reproduces the same answer:

```swift
// Equivalent expansion: invert, then apply the inverse to b
let x = try b.transformedBy(A.inverted())
// [1.0, 3.0]
```

The `solve(_:)` method computes `x = A⁻¹ · b` internally. The two forms are the same arithmetic wearing different clothes. They differ only in how they report failure. The compact call returns `nil` when the inverse does not exist. The expanded form throws `MatrixError.singular` from the inversion — the same failure by a different route. The next section is about when that happens.

### When there is no unique solution

Two lines cross at one point only when they point in different directions. Sometimes the second equation carries no new information. Then the matrix `A` is **singular** — a matrix with no inverse and no single answer to return. The plain cause is that the rows of `A` are **linearly dependent** — one row is a scalar multiple or a combination of the others. That row adds nothing the earlier rows did not already say — the mark of linear dependence. The <doc:Linear-Independence-Primer> takes up that property directly.

The café makes the failure concrete. Suppose the second receipt is lost, and in its place we find a duplicate order — exactly the first receipt, doubled:

```swift
// Receipt 2 is now 2 × Receipt 1 — linearly dependent
let dependent = [[2.0, 1.0],
                 [4.0, 2.0]]
```

A zero determinant is the exact-arithmetic signature of this collapse. The collapse wears two faces. In the first face, the dependent equations agree with each other. The doubled receipt shows a doubled total, so it describes the same line of possible prices twice, and every point on that line solves the system. This is the *redundant* case with infinitely many solutions. In the second face, the dependent equations disagree — the doubled order somehow rang up a different total. Same slope, different intercept: parallel lines that never meet. This is the *contradictory* case with no solution, and in the café it means one of the receipts is simply wrong. Both faces share the same singular matrix:

```swift
// Redundant: doubled order, doubled total (same line twice)
dependent.solve([5.0, 10.0])  // nil — infinitely many price pairs fit

// Contradictory: doubled order, incompatible total (parallel lines)
dependent.solve([5.0, 11.0])  // nil — no price pair fits
```

Redundant and contradictory are the two faces of a zero determinant — not two new kinds of matrix. Either way `solve(_:)` returns `nil`. Neither face yields the one answer the method promises. Quiver does not tell the two apart. A `nil` return means no unique answer, and it stays silent on which of the two reasons produced that result.

### Almost dependent, almost useless

Exact duplication is the clean case. The dangerous one is a receipt that is *almost* a multiple of another — a total misrecorded by a penny, an item count off by one. Suppose the doubled receipt's muffin count was entered as `2.01` instead of `2`:

```swift
let smudged = [[2.0, 1.0],
               [4.0, 2.01]]
smudged.solve([5.0, 10.0])   // Optional([2.5, 0.0]) — coffee $2.50, muffins free?
```

The system is technically solvable — the rows are not *quite* dependent — and the answer is nonsense. A one-cent smudge in the data swung coffee from `$1.00` to `$2.50` and priced muffins at nothing. This is the grey zone between a clean solve and an outright singular matrix, and the **condition number** is the diagnostic that measures it. See the <doc:Determinants-Primer> for the condition number in depth, including where to check it before trusting an answer.

### The feature-matrix connection

The same collapse appears in machine learning, where it has its own name. Sometimes a feature matrix has one column that is a linear combination of the others. Those features are **collinear** — linear dependence read across the columns of a data table rather than the rows of an equation system. Temperature recorded in both Celsius and Fahrenheit is the textbook case. The second column carries no information the first did not.

Collinear features make the matrix a model must invert singular. Fitting a linear regression on redundant features fails the same way solving a redundant system does. See the <doc:Determinants-Primer> for how this traces through the normal equation that ``LinearRegression`` solves. The <doc:Model-Interpretation-Primer> reads the condition number and the coefficients to catch collinearity before and after a fit.

### How Quiver decides

It helps to separate three related ideas that are easy to conflate. The determinant carries the *meaning*. A determinant of zero is the exact-arithmetic definition of a singular matrix. The *code* decides differently. It does not compute a determinant and test it against zero. Instead `inverted()` runs Gaussian elimination and watches the pivots. When a pivot falls below a small threshold, the method reports the matrix as singular. The determinant is a separate computation on its own path. It is not the gate the solver checks.

Between a clean solve and an outright singular matrix lies the grey zone the smudged receipt landed in — a matrix whose determinant is tiny but not zero. Such a matrix sits close enough to singular that the answer it returns is numerically unreliable. The determinant alone cannot flag this case. A small nonzero value still passes a zero test. The **condition number** is the diagnostic that measures it — a single number that quantifies how far a matrix sits from the singular boundary.

> Note: The distinction is worth holding onto. A zero determinant is what singular *means*. A vanishing pivot is how the solver *detects* it. The condition number is what measures the unreliable middle ground the determinant cannot see.

### Where to go from here

Solving a system is one application of matrix inversion, and the `nil` we met along the way has a cause worth naming: linear dependence among the rows. The <doc:Linear-Independence-Primer> takes up that property in its own right — whether the rows carry genuinely independent information. The <doc:Determinants-Primer> follows it with the scalar test for that property, running from determinants to invertibility to the condition number. The <doc:Linear-Algebra-Primer> steps back to the vectors and transformations that give `Ax = b` its geometry. The <doc:Machine-Learning-Primer> follows the collinearity thread into the models that solve systems to fit their coefficients.

> Experiment: **The Quiver Notebook** is the right place to watch a solvable system slide into an unsolvable one. Start with the café's `[[2, 1], [1, 3]]` system and print `solve([5, 10])`. Then edit the second row toward a double of the first — `[4, 2.1]`, then `[4, 2.01]`, then `[4, 2]`. Print `solve([5, 10])`, `determinant`, and `conditionNumber` at each step. The prices stay sensible, then swing wildly, then vanish to `nil`. Watching that slide is the fastest way to feel where the unique answer lives. See <doc:Quiver-Notebook>.
