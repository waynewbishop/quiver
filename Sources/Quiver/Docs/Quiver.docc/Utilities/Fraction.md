# Fraction

Representing rational numbers as exact fractions for clear display of matrix and vector results.

## Overview

Floating-point numbers are practical for computation but unkind to readers. The decimal `0.384615384615...` says nothing about where the value came from, while the same value written as `5/13` reveals an entire structure — a denominator that ties straight back to the determinant of the matrix it came from. `Fraction` is Quiver's presentation-only type for that translation.

A `Fraction` carries an integer `numerator` and a positive integer `denominator`. The two together describe a rational number in lowest terms. This type is not used in calculation — every operation in Quiver continues to return `Double` and `[[Double]]` — but is reached for whenever the rational structure of a result is itself the lesson. The <doc:Determinants-Primer> uses `asFractions()` to show how every entry of an inverted matrix shares the determinant as a denominator. The <doc:Vector-Operations> page uses `asFractions()` on the unit vector `[0.6, 0.8]` to recover `[3/5, 4/5]` and connect normalization back to a 3-4-5 triangle.

### Constructing a Fraction

The most common entry point is the `asFraction()` extension on `Double`, which converts a single floating-point value into its simplest rational form. For an explicit construction from a known numerator and denominator, the initializer reduces the result to lowest terms:

```swift
import Quiver

// From a Double — derives the rational form
let third = (1.0 / 3.0).asFraction()
print(third)            // "1/3"
print(third.numerator)  // 1
print(third.denominator)// 3

// Explicit numerator and denominator — reduced automatically
let half = Fraction(numerator: 6, denominator: 12)
print(half)             // "1/2"
```

The conversion uses a continued-fraction search through the Stern-Brocot tree, with a default `maxDenominator` of `1000`. Pass a larger ceiling when the source value is known to be a fraction with an unusually large denominator.

### Working with arrays of fractions

The same conversion lifts naturally to arrays. The `asFractions()` method on `[Double]` returns `[Fraction]`, and the matrix overload on `[[Double]]` returns `[[Fraction]]`. The matrix form is the one that pays for the type — it turns an inverted matrix into something a reader can immediately interpret:

```swift
let A = [[3.0, 1.0],
         [2.0, 5.0]]

let inverse = try A.inverted()
inverse.asFractions()
// [[5/13, -1/13],
//  [-2/13, 3/13]]

A.determinant.asFraction()  // 13
```

Every entry shares `13` as the denominator because the determinant is `13`. The decimal form of the same matrix hides that relationship behind `0.384615...` and `-0.076923...`.

### Equality and serialization

`Fraction` conforms to `Equatable`, which makes it directly comparable in tests and unit checks — two fractions are equal when their reduced numerator and denominator agree. It conforms to `CustomStringConvertible` so that `print()` and string interpolation render the familiar `"a/b"` form, falling back to `"a"` for whole numbers. It conforms to `Sendable` so that values produced on one task can be passed safely across concurrency boundaries without further annotation.

> Note: `Fraction` is presentation-only. Every Quiver operation still computes on `Double` and `[[Double]]`. Reach for it when the goal is to display a result, not to extend computation into rational arithmetic.

## See also

- <doc:Determinants-Primer>
- <doc:Vector-Operations>
- <doc:Matrix-Operations>
