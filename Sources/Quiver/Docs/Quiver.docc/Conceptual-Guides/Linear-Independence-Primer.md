# Linear Independence

Understand when a set of vectors carries genuinely new information, and when one of them is redundant.

## Overview

In linear algebra, a vector represents a movement or a direction in space. A matrix is built from a collection of these vectors. As we add vectors together, some of those vectors add a genuinely new direction. Others merely repeat what the earlier ones already reach. 

When no vector in a set can be built by scaling and adding the others together (a process known as taking a linear combination), the set is **linearly independent**. When a vector can be formed this way, the set is **dependent**, and the space it describes quietly collapses — a plane crushed flat into a line.

We build the idea one vector at a time, the way we would build a coordinate system.

> Note: In the previous primer we watched `solve(_:)` return `nil` when a system had no unique answer. This primer explains the property behind that failure. It builds on the vectors and `cosineOfAngle(with:)` from the <doc:Linear-Algebra-Primer> and on the `solve(_:)` we already met in the <doc:Solving-Systems-Primer>.

### One vector reaches a line

Start with a single vector b₁. Scaling it by every possible number — doubling it, halving it, flipping it negative — reaches every point on one line through the origin. That line is everything b₁ alone can reach:

```swift
import Quiver

let b1 = [1.0, 3.0]
// Every multiple of b1 lies on the same line:
// [2.0, 6.0], [0.5, 1.5], [-1.0, -3.0] ...
```

The set of points a collection of vectors can reach through scaling and adding is its **span**. The span of one non-zero vector is a single line. One vector alone cannot be dependent — there is nothing yet for it to depend on. The question of independence begins with the second vector.

### A second direction, or the same one again

Add a second vector and ask one question: does it supply a genuinely different direction, or does it merely retrace b₁'s line? For two vectors, "dependent" can only mean one thing — one is a scalar multiple of the other — and we already own the instrument that detects a shared line. Parallel vectors have a `cosineOfAngle` of exactly `1.0` (or `-1.0` when they point opposite ways):

```swift
let b2 = [2.0, 1.0]
b1.cosineOfAngle(with: b2)   // 0.707 — not ±1, a genuinely new direction

let b3 = 2.0 * b1            // [2.0, 6.0]
b1.cosineOfAngle(with: b3)   // 1.0 — the same line wearing a longer arrow
```

The pair b₁ and b₂ is **linearly independent**: neither is a multiple of the other, so each contributes a direction the other cannot reach. Their span is no longer a line but the entire plane — some number of b₁'s plus some number of b₂'s reaches any point in two dimensions. The pair b₁ and b₃ is **dependent**: b₃ was manufactured as `2.0 * b1`, so it arrives already spoken for, and the span of the pair is still just b₁'s original line.

### The combination test

The cosine answers the two-vector case. The general definition covers any number of vectors, and it is the one worth memorizing. A vector is dependent on a set when it can be written as a **linear combination** of them — a scaled sum. For a candidate third vector, the question is whether numbers a₁ and a₂ exist that satisfy:

`b₃ = a₁ · b₁ + a₂ · b₂`

If such numbers exist, b₃ adds nothing — it lands where b₁ and b₂ already reach. If no such numbers exist, b₃ is independent of them and contributes a new direction.

To find out whether such numbers exist, we hand the search to `solve(_:)`, the method we already used in the <doc:Solving-Systems-Primer>. Place b₁ and b₂ into a matrix as columns, give it the target, and the coefficients it returns are the combination — the a₁ and a₂ we were looking for:

```swift
// Which combination of b1 and b2 reaches [4, 7]?
// a1 · [1, 3] + a2 · [2, 1] = [4, 7]
let columns = [[1.0, 2.0],
               [3.0, 1.0]]     // b1 and b2, written as columns
columns.solve([4.0, 7.0])      // Optional([2.0, 1.0]) — a1 = 2, a2 = 1
```

Two b₁'s plus one b₂ lands exactly on `[4, 7]`. And because b₁ and b₂ are independent in a two-dimensional space, this search succeeds for *every* target — the pair spans the whole plane. That has a consequence worth pausing on: in the plane, any third vector is automatically dependent, because the plane is already full. A genuinely new third direction has to leave the plane entirely — a component out of the page, into a third dimension. Independence and dimension are two views of the same count.

Now run the same search on the dependent pair. Both columns point down one line, and no combination of copies of a single direction can ever leave that line:

```swift
// b1 and b3 share a line — can any combination reach [4, 7]?
let sameLine = [[1.0, 2.0],
                [3.0, 6.0]]    // b1 and b3 as columns
sameLine.solve([4.0, 7.0])     // nil — [4, 7] is off the line; no combination reaches it
```

That `nil` is what dependence looks like in code. Because the two columns share a line, they can only reach points on that line, and `[4, 7]` is not one of them — so no combination exists, and with no coefficients to return, `solve(_:)` answers `nil`. A returned pair means the vectors reached the target between them; a `nil` means they could not.

### Basis and dimension

The independent pair earns a name. A **basis** is a set of n linearly independent vectors whose span is the whole space, and the space is then **n-dimensional** — dimension is nothing more than the count of independent directions a basis holds. b₁ and b₂ form a basis for the plane. The familiar **natural basis** `[1, 0]` and `[0, 1]` is another basis for the same plane; a space has many bases, but every basis for it has the same count.

Basis vectors do not have to be unit length, and they do not have to sit at right angles to each other. b₁ and b₂ are neither. But everything is easier when they are: an **orthonormal** basis — mutually perpendicular unit vectors, like the natural basis — is the one where dot products and cosines behave most simply, so we reach for it whenever we have the choice.

### Redundancy is dependence in data

The same idea appears anywhere information repeats. Recall the duplicate café order from the <doc:Solving-Systems-Primer>, the pair that made `solve(_:)` return `nil` when we tried to price a coffee and a muffin. The first receipt recorded *two coffees and one muffin*, and the duplicate was that same receipt doubled — *four coffees and two muffins*, exactly twice the first — a linear combination. Two unknowns now rest on a single independent equation, and no unique pair of prices is forced. Nothing needs to be computed to see it: halve the doubled receipt and it repeats the first. Dependence in data is information that arrives already spoken for.

### Near dependence, and a warning shape

Exact dependence is rare in measured data. Near dependence is common, and the combination search shows what it does. Slide b₂ from its honest direction `[2.0, 1.0]` toward a near-multiple of b₁ and re-run the same search for `[4.0, 7.0]`:

```swift
// b2 now almost parallel to b1 = [1, 3]
let nearlyOneLine = [[1.0, 2.0],
                     [3.0, 5.9]]
nearlyOneLine.solve([4.0, 7.0])   // Optional([-96.0, 50.0])
```

The search still succeeds — the columns are not *quite* on one line — but look at the answer. Reaching a modest target now takes ninety-six negative copies of one vector fighting fifty copies of the other, two huge weights nearly cancelling. Nudge either column slightly and the pair swings just as violently somewhere else. This exploding, opposing pair is the signature of near-dependence: the target is still reachable, but only by having large, opposite contributions nearly cancel, and the answer is so sensitive that a tiny change in the data rewrites it completely.

### The geometry of collapse

One picture ties the sections together. A matrix's columns are where the basis vectors land: column one is the new home of `[1, 0]`, column two the new home of `[0, 1]`. Land them in different directions and they fence off a parallelogram with real area, and the transformed space keeps both of its dimensions. Land them on the same line and the parallelogram is flat. The basis vectors have become dependent, the plane is crushed onto a line, and the map cannot be undone — everything on that line traces back to a whole family of starting points. One collapse, visible three ways at once: a flattened parallelogram, a redundant receipt, and a failed combination search.

### Two meanings of independent

The word *independent* has an older home in statistics, where it means that two quantities carry no information about each other — knowing one tells you nothing about the other. That is a different idea from the linear independence defined here, which is about vectors and combinations alone. The two share a word and are easy to conflate, but they are separate properties: two quantities can move together and yet, as vectors, still point in genuinely different directions. Everywhere this primer and its linear-algebra neighbors say *independent*, they mean the vector sense.

### A word for what is lost

Basis and dimension make one more term precise. **Rank** is the dimension of the span — the count of independent directions a set of vectors actually delivers, as opposed to the count it appears to have. Two independent columns have rank `2` and span the plane: full rank, nothing collapsed. Make one column a multiple of the other and the rank drops to `1`; the pair still *looks* like two vectors, but it spans only a line. The café's duplicated receipt drops the rank of its system by one, and that missing direction is exactly the missing constraint.

The hand tools of this primer decide independence for a pair of vectors and for a square system, and the fully general test for any set of vectors is the rank of the matrix they form. Quiver exposes no `rank` property, so we read rank deficiency off its symptoms rather than off a single call. For a square system those symptoms are the ones dependence always shows there — a zero determinant, and a `solve(_:)` that returns `nil` where a combination should have been found.

### Where this leaves us

Two hand tools carried the whole idea: a cosine for a pair of vectors, and a combination search for anything larger. With nothing more than those, we defined independence, built span and basis and dimension on top of it, watched a dependent set collapse a plane onto a line, and named the rank that such a collapse destroys. That is the property in full, established with the two instruments this primer is built on.

These tools establish the property in full. The <doc:Determinants-Primer> adds a single number that reports it at a glance — a scalar test that reads zero when the columns collapse and non-zero when they hold their ground, folding this whole primer's work into one value.

> Experiment: **The Quiver Notebook** is the right place to watch independence fail one step at a time, using only this primer's two instruments. Fix `b1 = [1.0, 3.0]` and a target of `[4.0, 7.0]`, then sweep the second column from `[2.0, 1.0]` toward `[2.0, 6.0]` — try `[2.0, 4.0]`, `[2.0, 5.9]`, `[2.0, 5.99]`, `[2.0, 6.0]` — printing `cosineOfAngle(with:)` and the `solve(_:)` combination at each step. The cosine climbs toward `1.0`, the combination coefficients explode into ever-larger opposing pairs, and at exact dependence the search returns `nil`. One collapse, felt through the two instruments this primer is built on. See <doc:Quiver-Notebook>.
