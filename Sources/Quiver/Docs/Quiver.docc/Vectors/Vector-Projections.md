# Vector Projections

Decompose vectors into parallel and perpendicular components.

## Overview

Every vector can be split into two parts relative to any direction: the part that points along that direction and the part that points away from it. This decomposition is called projection, and it answers one of the most practical questions in applied mathematics: how much of this thing is going in the direction I care about?

### What projection measures

Imagine standing in sunlight. Your shadow on the ground is a projection — it shows how much of your height falls along the ground plane. A tall person casts a long shadow. A person lying flat casts a shadow equal to their full length. A person standing perpendicular to the ground casts no shadow at all.

Vector projection works the same way. Given a vector and a reference direction, the **scalar projection** measures how far the vector reaches along that direction, which is the length of the shadow. The **vector projection** gives that shadow as a vector, pointing along the reference direction with the measured length. Together they answer: "how much of this vector is aligned with that direction?"

![Vector b projected onto vector a, showing the parallel component along a and the orthogonal component perpendicular to it](diagram-vector-projection)

### What the orthogonal component measures

If the projection is the shadow, the orthogonal component is everything the shadow misses. It captures the part of the vector that points away from the reference direction. "Orthogonal" means perpendicular. The orthogonal component measures how far the vector strays from the direction we care about.

The two parts, parallel and perpendicular, always add back to the original vector. Nothing is lost, nothing is created. Projection is a lossless decomposition.

> Tip: Think of projection as asking two questions at once: "How much goes with?" (the projection) and "How much goes against?" (the orthogonal component). Any vector, any direction, any number of dimensions.

### Computing projections

Quiver provides three methods that work together. Consider a force vector and a ramp direction — we want to know how much force pushes along the ramp versus how much presses into its surface:

```swift
import Quiver

let force = [3.0, 4.0]
let ramp  = [2.0, 1.0]

// How far the force reaches along the ramp (a number)
let along = force.scalarProjection(onto: ramp)  // 4.47

// The force component parallel to the ramp (a vector)
let parallel = force.vectorProjection(onto: ramp)  // [4.0, 2.0]

// The force component perpendicular to the ramp (a vector)
let perpendicular = force.orthogonalComponent(to: ramp)  // [-1.0, 2.0]

// The two components reconstruct the original
let reconstructed = parallel.add(perpendicular)  // [3.0, 4.0]
```

The scalar projection is computed as v·u / |u| — the dot product divided by the magnitude of the reference vector. The vector projection scales the reference direction by that amount: proj_u(v) = (v·u / u·u) × u. The orthogonal component is whatever remains: v − proj_u(v).

> Important: The reference vector cannot be a zero vector. A zero vector has no direction, so projection onto it is undefined.

### Why projection matters

Projection appears anywhere a force, motion, or signal acts at an angle to a surface, path, or direction. The parallel component is the part that does useful work. The perpendicular component is the part that does not.

**Decomposing force on a ramp.** When pushing a crate up a ramp, only the component of force along the ramp moves the crate. The perpendicular component presses the crate into the surface — it creates friction but no movement:

```swift
import Quiver

// Push force: 100N at 30° above horizontal
let push = [86.6, 50.0]

// Ramp rises 3 meters over 4 meters horizontal
let ramp = [4.0, 3.0]

// Force that moves the crate up the ramp
let useful = push.vectorProjection(onto: ramp)

// Force wasted pressing into the ramp surface
let wasted = push.orthogonalComponent(to: ramp)

// Work done = magnitude of useful force × distance
let work = useful.magnitude * ramp.magnitude
```

This same decomposition applies anywhere a force acts at an angle: wind on a sail, gravity on a slope, thrust on an orbital trajectory.

**Reflecting off a surface.** When a ball bounces off a wall, the component along the surface normal reverses direction while the component along the surface stays the same. The reflection formula uses projection directly:

```swift
import Quiver

// Ball moving down-right toward a horizontal floor
let velocity = [3.0, -4.0]

// Floor normal points straight up
let normal = [0.0, 1.0]

// Reflect: reverse the normal component, keep the surface component
let normalPart = velocity.vectorProjection(onto: normal)
let reflected = velocity.subtract(normalPart * 2)  // [3.0, 4.0]
```

The formula `v − 2 × proj(v onto n)` works for any surface orientation in any number of dimensions. Game engines and ray tracers use this calculation on every frame.

**Course correction.** An aircraft has a velocity vector and a desired heading. The scalar projection onto the heading measures groundspeed in the right direction. The orthogonal component measures crosswind drift:

```swift
import Quiver

// Aircraft velocity: 200 knots, drifting 15° right of course
let velocity = [193.2, 51.8]

// Desired heading: due east
let heading = [1.0, 0.0]

// Groundspeed along the desired track
let groundspeed = velocity.scalarProjection(onto: heading)  // 193.2

// Lateral drift rate
let drift = velocity.orthogonalComponent(to: heading)  // [0.0, 51.8]
let driftRate = drift.magnitude  // 51.8 knots sideways
```

### Connection to linear regression

The normal equation used in linear regression — θ = (X'X)⁻¹X'y — is a projection. It projects the target vector onto the column space of the feature matrix. The result is the closest point in that space to the target, which is the least-squares best fit.

The intuition built here carries directly into that context. The prediction is the parallel component — the part of the target that the features can explain. The residual error is the orthogonal component — the part the features cannot reach. The best-fit model is the one where the error is perpendicular to every feature, meaning no feature can reduce it further.

Polynomial regression generalizes the same projection. `polyfit(x:y:degree:)` projects the target vector `y` onto the column space spanned by `[1, x, x², ..., xⁿ]` — a Vandermonde-style basis whose powers replace the raw features of linear regression. The geometry is identical; only the columns change.

For a full treatment of the normal equation and how Quiver solves it, see <doc:Linear-Regression>. For the `Polynomial` value type and `polyfit`, see <doc:Polynomials>.

## Topics

### Projection operations
- ``Swift/Array/scalarProjection(onto:)``
- ``Swift/Array/vectorProjection(onto:)``
- ``Swift/Array/orthogonalComponent(to:)``

### Related
- <doc:Linear-Algebra-Primer>
