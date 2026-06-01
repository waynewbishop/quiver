# Calculus Primer

The math behind derivatives and the iterative path to a minimum.

## Overview

A car on the highway covers 60 miles in one hour. Its speed is 60 miles per hour — and "miles per hour" is a **rate**, the same kind of measurement that shows up everywhere from a watch tracking pace on a run to a thermostat reporting how fast a room is cooling down. Every rate is the same shape: one quantity changes, another quantity tracks how fast it is changing. The math of those measurements is **calculus**, and the part of calculus this primer covers is the part Quiver actually uses.

> Note: This primer assumes no prior calculus background. A reader who has finished the <doc:Linear-Algebra-Primer> has everything they need to start here.

### A familiar rate

A walk up a hill produces elevation readings as we go. After the first second we have risen two meters. After the next, three more. Then four. Then five. The hill is getting steeper. The list of "how much we rose this second" is what calculus calls the rate of change of elevation over time.

```swift
import Quiver

let elevation = [100.0, 102.0, 105.0, 109.0, 114.0, 120.0]   // meters, one per second

let grade = elevation.derivative(sampleRate: 1.0)
// [2.0, 3.0, 4.0, 5.0, 6.0]
```

The output of `derivative(sampleRate:)` is the **derivative** of the elevation samples — the formal name for "how fast one quantity changes as another quantity changes." Each output is the rise from one second to the next. Speed is a derivative. Acceleration is a derivative of speed. Grade is a derivative of elevation. We have been computing derivatives all along; today we get the name.

> Note: `Array.derivative(sampleRate:)` uses the simplest possible numerical derivative — the difference between adjacent samples, divided by the time between them. When the underlying math is not written as a formula, the derivative is recovered from the data itself.

### When we know the formula

Sometimes the relationship between two quantities is known exactly, not just measured. Drop a ball off a roof. Its position below the roof, in meters, is `4.9 · t²` after `t` seconds. The `4.9` is half of Earth's gravitational acceleration; the `t²` is the textbook formula for distance fallen under gravity.

Knowing the formula, calculus produces a new formula — one that says how fast the ball is moving at any instant. We can also take the derivative of that formula, to get how fast the speed itself is changing:

> Note: `Polynomial` stores coefficients in ascending order — `[a₀, a₁, a₂]` represents `a₀ + a₁t + a₂t²`. The first element is the constant term, the second multiplies `t`, the third multiplies `t²`, and so on. See <doc:Polynomials> for the full type.

```swift
import Quiver

let position = Polynomial([0.0, 0.0, 4.9])    // 4.9t² — coefficients low-to-high
let velocity = position.derivative()           // 9.8t
let acceleration = velocity.derivative()       // 9.8 (constant)
```

Three formulas, two derivatives. The pattern that turns `4.9t²` into `9.8t` is the **power rule**: the exponent drops down as a multiplier (the `2` came down and turned `4.9` into `9.8`), and the new exponent is one smaller (`t²` became `t¹`, which we just write as `t`). The next derivative does the same again — the exponent on `t¹` was 1, it came down, and `t` itself disappeared (because anything raised to the zero power is 1). What is left is the constant `9.8`.

Because the coefficients are stored low-to-high, that exponent is also the slot the coefficient sits in: `4.9` lives at index 2, the same as the power on `t²`. So the power rule reads off the array directly — multiply each coefficient by its own index and shift it down one slot. The `4.9` at index 2 becomes `4.9 × 2 = 9.8` at index 1, and the constant at index 0 multiplies by zero and disappears, which is the same reason a constant has a derivative of zero.

A constant has a derivative of zero. The number `9.8` does not contain `t`; it does not change as time passes. Gravity is the same at every instant of the fall, so the rate of change of speed is the same number forever.

```swift
position(2.0)        // 19.6 — meters fallen after 2 seconds
velocity(2.0)        // 19.6 — meters per second at that instant
acceleration(2.0)    // 9.8  — meters per second squared, at every instant
```

Three numbers, three meanings, one formula behind all of them. That is what `Polynomial.derivative()` does — applies the power rule to every term and returns a new polynomial.

### Two faces of the same idea

The same word — derivative — names two operations in Quiver, one for each of the two situations where rates of change come up.

When we know the formula, the derivative is another formula:

```swift
let p = Polynomial([0.0, 3.0, 2.0])   // 2x² + 3x
let dp = p.derivative()                // 4x + 3
dp(5.0)                                // 23.0 — the rate of change at x = 5
```

When we only have samples, the derivative is more samples:

```swift
let temps = [68.0, 70.0, 73.0, 77.0, 80.0]     // °F, one reading per minute
let warmingRate = temps.derivative(sampleRate: 1.0)
// [2.0, 3.0, 4.0, 3.0] — degrees per minute
```

Same idea, different inputs. This is the same teaching move <doc:Linear-Algebra-Primer> makes when it pairs `magnitude` (distance from the origin) with `distance(to:)` (distance between two points) — one operation in two framings, chosen by what the caller has on hand.

> Note: The numerical derivative returns one fewer sample than the input array. Every difference needs two adjacent values, so a 100-sample input gives a 99-sample derivative.

### Finding the lowest point

So far the derivative has told us how fast something is changing. It can also tell us something more powerful, and this is the reason calculus shows up in machine learning: where the lowest point of a formula is.

Think of an error formula — a formula that says how wrong a model's predictions are on a given dataset. The model has knobs we can turn (the coefficients of a regression line, for example). Turning the knobs changes the error. Somewhere, there is a setting of those knobs that makes the error as small as it can be. That setting is the **minimum** of the error formula, and it is the answer the model wants.

The connection to derivatives is this: at a minimum, the rate of change is zero. The error formula has stopped going down — it has reached its lowest point — and from there, any small turn of a knob in either direction makes the error go up again. So to find the minimum, take the derivative of the error formula, and find the spot where the derivative equals zero.

### The closed-form answer

For one particular error formula — the squared error used by ordinary linear regression — this calculation produces a clean answer in a single matrix expression:

θ = (X'X)⁻¹X'y

This is the **normal equation**, and it is what falls out when we take the derivative of squared error, set it equal to zero, and solve for the coefficients θ. The derivation does not need to be followed line by line; the point is that `LinearRegression.fit` is not magic. It is the answer calculus gives to "where is the line with the least total error":

```swift
import Quiver

let sqft   = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
let prices = [150000.0, 200000.0, 260000.0, 310000.0, 370000.0]

let model = try LinearRegression.fit(features: sqft, targets: prices)
// LinearRegression: 1 feature, intercept: 38000.00, slope: 110.00
```

Every regression fit is a calculus problem solved in one step.

### When the answer can only be walked

Not every error formula has a clean answer. For some, taking the derivative and setting it equal to zero produces an equation that cannot be solved by algebra — there is no expression that says "here is the answer."

For those, the derivative still does something useful. It tells us which direction the minimum lies in. Not the exact spot, but the way to walk from where we are. So instead of solving the equation, we walk: start anywhere, look at the derivative to see which way is downhill, take a small step in that direction, and look again. Repeat until the derivative is essentially zero — meaning the ground is flat and we have arrived at the lowest point.

That iterative walk has a name. It is called **gradient descent**, and it is the algorithm Quiver's `GradientDescent` implements:

```swift
import Quiver

let model = try GradientDescent.fit(features: scaled, targets: targets)
// GradientDescent: 4 features, converged in 87 iterations (loss: 0.2530)
```

Linear regression's closed form is the case where calculus reaches the answer in a single matrix expression. Gradient descent is the case where calculus reaches the answer over many small ones. Both are calculus. The difference is whether the math can be solved directly or only followed step by step.

This matters now because the models that come after linear regression — logistic regression and the support vector machines beyond it — minimize error formulas for which no closed form exists. There is no normal equation for them. The only way to fit them is iteratively. The optimizer Quiver just introduced is the engine the next models will need.

### From calculus to optimization

Calculus runs through Quiver in four places. `Polynomial.derivative()` returns the derivative of a known formula. `Array.derivative(sampleRate:)` returns the derivative of a list of samples. `LinearRegression.fit` uses calculus to find the line of least squared error in one step. `GradientDescent` uses calculus to walk to a minimum when no closed form exists.

The same idea, four shapes. A derivative tells a model how fast something is changing, which direction is downhill, and when the ground is flat. That is everything calculus does here, and it is everything the models that follow will need.

> Experiment: **The Quiver Notebook** is the right place to watch a derivative show up as data. Take a polynomial that models something physical — `Polynomial([0.0, 0.0, 4.9])` for a falling object, or build one from sample data with `polyfit(x:y:degree:)` — then sample its derivative across an interval and plot both side by side. Watching the curve's slope become a second curve is what makes the rate-of-change idea concrete. See <doc:Quiver-Notebook>.
