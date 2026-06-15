# Calculus Primer

The math behind derivatives and the iterative path to a minimum.

## Overview

Calculus helps us measure how fast things change so we can better understand our models. We use this math to find the exact configuration that minimizes error. This
  approach provides the foundation for fitting models in Quiver.

  Calculus serves two roles in our work. In closed-form models like <doc:Linear-Regression> we use the derivative to solve for the best possible coefficients in a single step. For iterative models like <doc:Logistic-Regression> we use the derivative to walk down a slope until we reach the lowest point of error. This mathematical framework allows us to fit models on the device. It ensures we can measure the sensitivity of our predictions and verify the stability of our results regardless of how we find the solution.

> Note: This primer assumes no prior calculus background. A reader who has finished the <doc:Linear-Algebra-Primer> has everything they need to start here.

### A familiar rate

A walk up a hill produces elevation readings as we go. After the first second we have risen two meters. After the next, three more. Then four. Then five. Then six. The hill is getting steeper. The list of "how much we rose this second" is what calculus calls the rate of change of elevation over time.

```swift
import Quiver

let elevation = [100.0, 102.0, 105.0, 109.0, 114.0, 120.0]   // meters, one per second

let grade = elevation.derivative(sampleRate: 1.0)
// [2.0, 3.0, 4.0, 5.0, 6.0] — 102−100, 105−102, 109−105, 114−109, 120−114
```

The output of `derivative(sampleRate:)` is the **derivative** of the elevation samples: the formal name for "how fast one quantity changes as another quantity changes." Each output is the rise from one second to the next, which is why the result has one fewer value than the input: each grade is the change between two readings, and six readings leave five changes. Speed is a derivative. Acceleration is a derivative of speed. Grade is a derivative of elevation. We have been computing derivatives all along; today we get the name.

> Note: `Array.derivative(sampleRate:)` uses the simplest possible numerical derivative: the difference between adjacent samples, divided by the time between them. When the underlying math is not written as a formula, the derivative is recovered from the data itself.

### When we know the formula

Sometimes the relationship between two quantities is known exactly, not just measured. Drop a ball off a roof. Its position below the roof, in meters, is `4.9 · t²` after `t` seconds. The `4.9` is half of Earth's gravitational acceleration; the `t²` is the textbook formula for distance fallen under gravity.

Knowing the formula, calculus produces a new formula: one that says how fast the ball is moving at any instant. We can also take the derivative of that formula, to get how fast the speed itself is changing:

> Note: ``Polynomial`` stores coefficients in ascending order: `[a₀, a₁, a₂]` represents `a₀ + a₁t + a₂t²`. The first element is the constant term, the second multiplies `t`, the third multiplies `t²`, and so on. See <doc:Polynomials> for the full type.

```swift
import Quiver

let position = Polynomial([0.0, 0.0, 4.9])    // 4.9t² — coefficients low-to-high
let velocity = position.derivative()           // 9.8t
let acceleration = velocity.derivative()       // 9.8 (constant)
```

Three formulas, two derivatives. The pattern that turns `4.9t²` into `9.8t` is the **power rule**: the exponent drops down as a multiplier (the `2` came down and turned `4.9` into `9.8`), and the new exponent is one smaller (`t²` became `t¹`, which we just write as `t`). The next derivative does the same again: the exponent on `t¹` was 1, it came down, and `t` itself disappeared (because anything raised to the zero power is 1). What is left is the constant `9.8`.

Because the coefficients are stored low-to-high, that exponent is also the slot the coefficient sits in: `4.9` lives at index 2, the same as the power on `t²`. So the power rule reads off the array directly: multiply each coefficient by its own index and shift it down one slot. The `4.9` at index 2 becomes `4.9 × 2 = 9.8` at index 1, and the constant at index 0 multiplies by zero and disappears, which is the same reason a constant has a derivative of zero.

A constant has a derivative of zero. The number `9.8` does not contain `t`; it does not change as time passes. Gravity is the same at every instant of the fall, so the rate of change of speed is the same number forever.

```swift
position(2.0)        // 19.6 — meters fallen after 2 seconds
velocity(2.0)        // 19.6 — meters per second at that instant
acceleration(2.0)    // 9.8  — meters per second squared, at every instant
```

Three numbers, three meanings, one formula behind all of them. That is what `Polynomial.derivative()` does: it applies the power rule to every term and returns a new polynomial.

> Tip: The power rule reads straight off the coefficient array. Fit `polyfit(x:y:degree:)` to points on `y = 2x² + 3x + 1` and the coefficients come back as `[1, 3, 2]`: constant, then `x`, then `x²`. Call `.derivative()` and the result is `[3, 4]`: the `2` at index 2 became `4` at index 1, the `3` stayed as `3` at index 0, and the constant `1` dropped away. Each coefficient multiplied by its own index, shifted down one slot.

### Two faces of the same idea

The same word, derivative, names two operations in Quiver, one for each of the two situations where rates of change come up.

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

Same idea, different inputs. This is the same teaching move <doc:Linear-Algebra-Primer> makes when it pairs `magnitude` (distance from the origin) with `distance(to:)` (distance between two points): one operation in two framings, chosen by what the caller has on hand.

> Note: The numerical derivative returns one fewer sample than the input array. Every difference needs two adjacent values, so a 100-sample input gives a 99-sample derivative.

> Experiment: **The Quiver Notebook** is the right place to watch a derivative show up as data. Take a polynomial that models something physical (`Polynomial([0.0, 0.0, 4.9])` for a falling object), then sample its derivative across an interval and plot both curves side by side. Watching the curve's slope become a second curve is what makes the rate-of-change idea concrete. See <doc:Quiver-Notebook>.

### Finding the lowest point

So far the derivative has told us how fast something is changing. The derivative can also tell us something more powerful, and this is the reason calculus shows up in machine learning: where the lowest point of a formula is.

Think of an error formula: a formula that says how wrong a model's predictions are on a given dataset. The model has knobs we can turn (the coefficients of a regression line, for example). Turning the knobs changes the error. Somewhere, there is a setting of those knobs that makes the error as small as it can be. That setting is the **minimum** of the error formula, and it is the answer the model wants.

Being exact about what **error** means matters here, because the word carries the whole idea. The error is a single number measuring how far the model's predictions fall from the actual values: the gap between guess and truth, squared so that overshooting and undershooting both count as wrong and large misses count for more, then averaged across the dataset. This averaged squared error is the quantity that code and machine-learning writing usually call the **loss**; the two words name the same number, and the <doc:Gradient-Descent> model carries it step by step as its `lossHistory`. A large error means the predictions are far off; an error of zero means they match the data exactly. Squaring is also why the error formula is bowl-shaped: a squared quantity traces a parabola, a bowl with a single lowest point.

### Where the derivative is zero

The connection to derivatives is this: at a minimum, the rate of change is zero. The error formula has stopped going down, it has reached its lowest point, and from there any small turn of a knob in either direction makes the error go up again. So to find the minimum, take the derivative of the error formula, and find the spot where the derivative equals zero.

Two different quantities are in play here, and keeping them apart is the key to reading any of this. The **error** is how wrong the model is: picture it as altitude, the height of the bowl at the current knob setting. The **derivative of the error** is the slope underfoot: which way the ground tilts and how steeply. They are not the same thing: one is the height of the landscape, the other is its tilt. Both reach zero at the minimum (the error is at its lowest, and the ground is flat), but they measure different things, and a slope near zero is the signal that the bottom is close, not the value of the error itself.

### The closed-form answer

For one particular error formula, the squared error used by ordinary linear regression, this calculation produces a clean answer in a single matrix expression:

θ = (X'X)⁻¹X'y

This is the **normal equation**, and it is what falls out when we take the derivative of squared error, set it equal to zero, and solve for the coefficients θ. The derivation does not need to be followed line by line; the point is that `LinearRegression.fit` is not magic. The fit is the answer calculus gives to "where is the line with the least total error":

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

For those, the derivative still does something useful. The slope tells us which direction the minimum lies in. Not the exact spot, but the way to walk from where we are. So instead of solving the equation, we walk: start anywhere, look at the derivative to see which way is downhill, take a small step in that direction, and look again. Repeat until the derivative is essentially zero, meaning the ground is flat and we have arrived at the lowest point.

A model has more than one coefficient to adjust, so there is more than one direction to consider. The derivative with respect to a single coefficient, holding the others fixed, is its **partial derivative**: the slope along that one coefficient's axis. Collecting the partial derivative for every coefficient into a list gives the **gradient**, the combined downhill direction the walk follows. Each step moves every coefficient a small amount along its own partial derivative at once.

That iterative walk has a name. The walk is called **gradient descent**, and it is the algorithm Quiver's ``GradientDescent`` implements:

```swift
import Quiver

// Four features whose targets follow the exact rule y = a + 2b + 3c + 4d.
let rawFeatures: [[Double]] = [[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0], [0, 0, 0, 1],
                               [1, 1, 0, 0], [0, 1, 1, 0],
                               [0, 0, 1, 1], [1, 1, 1, 1]]
let targets = [1.0, 2, 3, 4, 3, 5, 7, 10]

// Standardize, then let the optimizer walk to the minimum.
let scaler = StandardScaler.fit(features: rawFeatures)
let scaled = scaler.transform(rawFeatures)

let model = try GradientDescent.fit(features: scaled, targets: targets, learningRate: 0.1)
// GradientDescent: 4 features, converged in 174 iterations (loss: 0.0000)
```

Linear regression's closed form is the case where calculus reaches the answer in a single matrix expression. Gradient descent is the case where calculus reaches the answer over many small ones. Both are calculus. The difference is whether the math can be solved directly or only followed step by step.

Gradient descent is a **first-order** method: it uses only the first derivative (the slope) to decide which way to walk. A second family of optimizers also uses the second derivative, the rate at which the slope itself changes, to judge how far to step as well as which way. Those second-order methods, Newton's method and the Taylor approximations behind it, converge in fewer steps on some problems, at the cost of computing and inverting the matrix of second derivatives. Quiver's optimizer is first-order throughout; the second-order family is a separate track it does not implement.

This matters because the models beyond linear regression (``LogisticRegression``, and the support vector machines that will follow it) minimize error formulas for which no closed form exists. There is no normal equation for them. The only way to fit them is iteratively. The optimizer introduced here is the one those models use: <doc:Logistic-Regression> runs this same descent on a cross-entropy loss.

### From calculus to optimization

Calculus runs through Quiver in four places. `Polynomial.derivative()` returns the derivative of a known formula. `Array.derivative(sampleRate:)` returns the derivative of a list of samples. `LinearRegression.fit` uses calculus to find the line of least squared error in one step. ``GradientDescent`` uses calculus to walk to a minimum when no closed form exists.

The same idea, four shapes. A derivative tells a model how fast something is changing, which direction is downhill, and when the ground is flat. That is everything calculus does here, and it is everything the models that follow will need. The inverse operation, accumulating the area under a curve to recover a total from a rate, appears as integration in the <doc:Physics-Primitives-Primer>, where a signal's samples are summed back into a quantity like distance or energy.

> Experiment: **The Quiver Notebook** is the right place to watch the derivative find a minimum. Fit ``GradientDescent`` to a small dataset and plot its `lossHistory` against the iteration count: the error falls steeply at first, then flattens toward zero as each step follows the slope downhill and the slope itself shrinks to nothing. The flattening is the derivative reaching zero: the bottom of the bowl. Fit ``LinearRegression`` to the same data to confirm both routes reach the same predictions, one walked and one solved in a single step. See <doc:Quiver-Notebook>.
