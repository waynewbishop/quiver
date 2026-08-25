# Eigenvalues and Eigenvectors

Finding the characteristic directions a transformation preserves and the stretch each one carries.

## Overview

When a transformation alters space most vectors are knocked off their original path and point in a new direction. However, a
few special directions stay perfectly aligned with their original line. They might get longer or shorter, but they do not rotate.

Finding these special directions and measuring how much they stretch or shrink is the "eigenproblem." The word **eigen** is German for "own" or "proper". An eigenvector is a transformation's own direction, one it keeps to itself, which is why mathematicians also call these the characteristic directions of the matrix. We call the special direction an eigenvector, and the amount it stretches or shrinks its eigenvalue.

We can understand this best by looking at the geometry before dealing with any algebra. The <doc:Matrix-Transformations> guide showed how a matrix moves space, and the <doc:Determinants-Primer> primer measured the area changes those moves produce. This primer asks another question. After the transformation
is complete, which vectors are still pointing exactly where they started?

### The vectors that keep their direction

A matrix transforms every vector in the plane at once. Picture a unit square centered at the origin: a vertical scaling stretches it into a rectangle twice as tall, and a horizontal shear tilts it into a leaning parallelogram. The square is a device for seeing many vectors at the same time — every point on its outline is the tip of some vector, and the transformation moves all of them together. Most vectors change direction when the plane distorts. The interesting ones are the vectors that do not.

Take three vectors — one horizontal, one vertical, one diagonal at 45° — and apply a vertical scaling that doubles heights. `transformedBy(_:)` shows where each one lands:

```swift
import Quiver

let scale = [[1.0, 0.0],
             [0.0, 2.0]]

[1.0, 0.0].transformedBy(scale)  // [1.0, 0.0] — unchanged
[0.0, 1.0].transformedBy(scale)  // [0.0, 2.0] — same direction, twice the length
[1.0, 1.0].transformedBy(scale)  // [1.0, 2.0] — knocked off its line
```

The horizontal vector is untouched: same direction, same length. The vertical vector still points straight up, but its length has doubled. The diagonal vector loses on both counts: its length grows from `√2 ≈ 1.41` to `√5 ≈ 2.24`, and its angle steepens past 45°, so it no longer lies on its starting line. Any vector other than the horizontal and vertical ones would have been thrown off its line the same way.

The horizontal and vertical vectors are characteristic of this particular transformation, and they are its eigenvectors. The horizontal one kept its length, so its eigenvalue is `1`. The vertical one doubled, so its eigenvalue is `2`. Quiver computes both at once:

```swift
let eigen = try scale.eigenDecomposed()
eigen.eigenvalues   // [2.0, 1.0] — descending
eigen.eigenvectors  // [[0.0, 1.0], [1.0, 0.0]]
```

Each row of `eigenvectors` is one eigenvector, paired with the eigenvalue at the same position: the vertical direction `[0.0, 1.0]` carries the stretch of `2`, and the horizontal direction `[1.0, 0.0]` carries the `1`. That is the entire 2D eigenproblem — find the vectors still lying on their original span, then measure how much their length changed.

### When an eigenvalue is negative

A reflection across the horizontal axis flips every height and leaves widths alone:

```swift
let reflection = [[1.0, 0.0],
                  [0.0, -1.0]]

[0.0, 1.0].transformedBy(reflection)  // [0.0, -1.0] — flipped, same line
```

The vertical vector comes out pointing down, but it still lies on the vertical line where it started. The line is preserved even though the vector reversed, so the direction still counts as an eigenvector, and its eigenvalue is `-1`: the stretch factor that means a flip. The eigenvalues here are `1` and `-1`, and their product is `-1` — the negative determinant the <doc:Determinants-Primer> primer reads as a mirror.

### A shear and a rotation

Two more classic transformations test whether the idea generalizes. Under a pure horizontal shear — pure meaning no added scaling or rotation, so the area is unchanged — the same three vectors tell a different story:

```swift
let shear = [[1.0, 0.5],
             [0.0, 1.0]]

[1.0, 0.0].transformedBy(shear)  // [1.0, 0.0] — unchanged
[0.0, 1.0].transformedBy(shear)  // [0.5, 1.0] — tilted off vertical
[1.0, 1.0].transformedBy(shear)  // [1.5, 1.0] — knocked off its line
```

Only the horizontal vector survives on its original line, with its length intact: one eigenvector direction, eigenvalue `1`. Every other vector tilts.

A rotation is the extreme case:

```swift
// 90° counterclockwise rotation
let rotation = [[0.0, -1.0],
                [1.0, 0.0]]

[1.0, 0.0].transformedBy(rotation)  // [0.0, 1.0] — a quarter turn off its line
```

Every vector in the plane turns the same quarter turn, so none stays on its starting line. A rotation by anything other than a half or full turn has no real eigenvectors at all — no direction in the plane is characteristic of it.

> Note: The shear and rotation matrices are not symmetric, and `eigenDecomposed()` accepts square, symmetric matrices only. It throws ``MatrixError/notSquare`` for a non-square input and ``MatrixError/notSymmetric`` for a square matrix that is not symmetric. That restriction is deliberate, and the next section is why.

### Why symmetric matrices behave

The shear had a single eigenvector direction and the rotation had none, so a general matrix makes a shaky foundation for what comes next. Symmetric matrices — the ones equal to their own transpose — are the well-behaved case. A symmetric matrix always has a full set of real eigenvalues, and its eigenvectors are always perpendicular to each other:

```swift
import Quiver

let symmetric = [[4.0, 2.0],
                 [2.0, 3.0]]

let eigen = try symmetric.eigenDecomposed()
eigen.eigenvalues   // [5.56, 1.44]
eigen.eigenvectors  // [[0.788, 0.615], [-0.615, 0.788]]

let v1 = eigen.eigenvectors[0]
let v2 = eigen.eigenvectors[1]
v1.dot(v2)          // 0.0 — perpendicular by construction
```

The two directions form a right angle: their dot product is `0.0`. Together they act as a new pair of axes, custom-fitted to the transformation. Along those axes the matrix does nothing but stretch: by `5.56` on the first, by `1.44` on the second. And the connection to the <doc:Determinants-Primer> primer falls out directly: the two eigenvalues multiply to `8.0`, exactly the determinant `(4 × 3) − (2 × 2)`. The matrix scales space by `5.56` along one axis and by `1.44` along the perpendicular one, so the area of any shape is multiplied by their product. The determinant is that same area factor, computed here as the product of stretches along the characteristic axes rather than from the matrix entries directly.

A companion fact rides along. The **trace**, the sum of the diagonal, equals the sum of the eigenvalues: `4 + 3` is `7`, and `5.56 + 1.44` is `7.0`. The determinant multiplies the stretch factors, and the trace adds them.

Either direction along an eigenvector's line is equally valid, since a flipped vector still lies on the same line. Quiver fixes each sign so the largest-magnitude element is positive, and repeated runs print identical vectors.

### The eigenvalues of a covariance matrix

The symmetric matrices this pays off for are the ones data produces. A covariance matrix — each feature's variance on the diagonal, each pairing's co-movement off it — is symmetric by construction, so its eigenvectors and eigenvalues always exist and always behave. Six workouts, described by duration in minutes and active energy in kilocalories:

```swift
import Quiver

// [duration, activeEnergy] — minutes and kilocalories per workout
let sessions = [
    [35.0, 385.0],
    [41.0, 295.0],
    [67.0, 790.0],
    [49.0, 510.0],
    [66.0, 715.0],
    [36.0, 495.0]
]

if let covariance = sessions.covarianceMatrix() {
    let eigen = try covariance.eigenDecomposed()
    eigen.eigenvalues  // [36102.26, 42.81]
}
```

Read through the geometric lens, these numbers say something concrete about the data. The top eigenvector is the direction in feature space along which the workouts vary most, and its eigenvalue, `36102.26`, is the variance found along it. The second, perpendicular direction holds what little variation is left, and dropping that axis would cost almost none of the data's spread. The eigenproblem has turned a table of numbers into a set of characteristic axes, ordered by how much of the data's spread each one carries.

### From characteristic directions to components

That reading — directions of variance, ranked by eigenvalue — is principal component analysis, one step before it gets its name. The <doc:Principal-Component-Analysis> model doc picks up exactly here: standardize the features so units stop dominating, keep the top directions, and project the data onto them. The <doc:Determinants-Primer> primer holds the other half of the story, where the determinant these eigenvalues multiply into measures invertibility and conditioning. And the pictures this primer leaned on are built in <doc:Matrix-Transformations>, one transformation at a time.

> Experiment: **The Quiver Notebook** is the right place to hunt eigenvectors by hand. Build the vertical scaling `[[1.0, 0.0], [0.0, 2.0]]`, transform a dozen unit vectors at different angles with `transformedBy(_:)`, and for each one compare its direction before and after. Every vector drifts toward the vertical except two — the axes themselves — because the vertical stretch of `2` exceeds the horizontal stretch of `1`, and the drift grows with the angle between a vector and its nearest eigenvector. See <doc:Quiver-Notebook>.
