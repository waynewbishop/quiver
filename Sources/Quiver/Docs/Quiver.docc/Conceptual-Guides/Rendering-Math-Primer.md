# Rendering Math

Reading a numeric result in the form mathematics uses, including fractions, column vectors, bracketed matrices, and descending polynomials.

## Overview

A textbook writes the inverse of a 2×2 matrix as a grid of fractions. Swift writes the same result as a grid of decimals. The math is identical; the form is not. Decimals are how a computer stores numbers; fractions, columns, brackets, and descending polynomials are how readers reason about them. **Rendering** is the bridge between the two — taking a `Double`, a `[Double]`, a `[[Double]]`, or a `Polynomial` and showing it in the form a reader can read.

Quiver builds the bridge with two complementary methods. `asFraction` returns rational structure as a real `Fraction` value, carrying a numerator, a denominator, and equality so the structure can be held and compared, not only printed. `asExpression` returns a Unicode-formatted string ready to display in the Notebook output pane, the Xcode console, or a SwiftUI `Text` view styled with a monospaced font. The two compose. Chaining `asFractions().asExpression()` on a matrix shows the rational form of every cell, stacked in textbook brackets. This primer walks the family one concept at a time and closes on what the rendering shows when the underlying floating-point arithmetic has left a trace.

> Note: This primer pairs with <doc:Numerical-Literacy>. The rendering rules described here are deliberate about presenting `NaN`, `±∞`, and machine-noise coefficients so the reader sees them. The numerical-literacy primer explains what those signals mean.

### From decimal to fraction

Every Quiver result computes in `Double`. When the decimal form hides structure worth seeing, `asFraction` recovers the rational form:

```swift
import Quiver

let ratio = 1.0 / 3.0

ratio.asFraction()              // 1/3 — a Fraction value
ratio.asFraction().numerator    // 1
ratio.asFraction().denominator  // 3
ratio.asFraction().value        // 0.3333333333333333 — back to Double
```

The return type matters. `Fraction` is a real value type, not a string, so two fractions can be compared, used as dictionary keys, or encoded through `Codable`. The structure is held, not only printed. When the rendered form is all that is needed, chain into `asExpression`:

```swift
ratio.asFraction().asExpression()   // "1/3"
```

The asymmetry is deliberate. Quiver returns *structure* when there is structure worth keeping and returns *text* when the rendering is itself the artifact. The chain joins the two without forcing a closure on the calling code.

### Vectors as columns

A vector and a column of numbers are the same mathematical object. The column form makes that identity visible at a glance, which is what a rendered expression is for. When a matrix multiplies a vector, the vector enters as a column. When a model returns a gradient, the gradient is a column. When any linear-algebra reference shows a vector, the vector appears as a column. The default matches the math:

```swift
[3.0, 4.0, 5.0].asExpression()
// ⎡ 3 ⎤
// ⎢ 4 ⎥
// ⎣ 5 ⎦
```

For places where a stacked column would interrupt the flow of prose, such as a sentence in a doc comment or one line of a debug log, the `.inline` form returns the angle-bracket variant:

```swift
[3.0, 4.0, 5.0].asExpression(form: .inline)   // "⟨3, 4, 5⟩"
```

Two short cases round out the contract. A single-element vector renders as the scalar itself, because a one-cell column is awkward where the scalar form reads cleaner. An empty vector renders as `⟨⟩` in either form, a deliberate marker (distinct from any numeric value) that the vector exists but holds no entries:

```swift
[5.0].asExpression()             // "5" — the scalar form, no brackets
[Double]().asExpression()        // "⟨⟩" — empty marker
```

### Matrices in textbook form

Matrices render as a bracketed grid with each column right-aligned to its widest cell, so negative signs and decimal points line up vertically:

```swift
let A: [[Double]] = [[3, 1], [2, 5]]

A.asExpression()
// ⎡ 3  1 ⎤
// ⎣ 2  5 ⎦
```

The alignment is what makes a column of mixed-sign numbers readable. A `-2` and a `3` sit in the same column with the minus sign in its own slot, not crowding the digit. Two shape rules cover the corners. A single-row matrix collapses to inline form `[ 1  2  3 ]`. The stacked Unicode brackets only earn their weight when the matrix has more than one row, so the single-row case uses square brackets. An empty matrix renders as `⟨⟩`, the same marker the empty vector uses.

### Polynomials in reading order

Quiver stores polynomial coefficients in **ascending order of power**, with `coefficients[0]` as the constant term, `coefficients[1]` as the linear term, and so on. Ascending storage is what arithmetic operations produce naturally and what numerical solvers consume. Readers, on the other hand, write polynomials in descending order, highest power first. The `asExpression` rendering reverses storage to match the reader:

```swift
let p = Polynomial([1, 3, 2])   // a₀=1, a₁=3, a₂=2 — stored ascending

p.coefficients                  // [1.0, 3.0, 2.0]
p.asExpression()                // "2x² + 3x + 1" — rendered descending
```

Reading the rendering against the storage walks the convention. Storage index `0` holds `1`, the constant term, so `1` appears with no `x`. Index `1` holds `3`, so `3x` is the linear term. Index `2` holds `2`, so `2x²` is the quadratic term. The rendering reverses the iteration order: highest-degree first, lowest last. Storage stays as the solver writes it; the printed form matches how the polynomial is written by hand.

Three formatting conventions hold across the rendering. Coefficients of `0` are omitted entirely. Coefficients of `±1` on `x` or higher powers drop the leading digit, so `x²` appears rather than `1x²`. Negative terms join with ` - ` rather than ` + -`, so the printed expression reads the way it would be written:

```swift
Polynomial([0, -1]).asExpression()   // "-x" — unary minus, no leading space
Polynomial([0]).asExpression()       // "0" — the zero polynomial
```

See <doc:Polynomials> for the type itself, including evaluation, differentiation, and least-squares fitting.

### Chaining structure into display

Decimal output and rational output answer different questions. A 2×2 inverse computed in decimals looks like four unrelated numbers. The same inverse, rendered as fractions, reveals that every entry shares the determinant as its denominator:

```swift
import Quiver

let A: [[Double]] = [[3, 1], [2, 5]]
let inverse = try A.inverted()

inverse.asExpression()
// ⎡  0.3846  -0.0769 ⎤
// ⎣ -0.1538   0.2308 ⎦

inverse.asFractions().asExpression()
// ⎡  5/13  -1/13 ⎤
// ⎣ -2/13   3/13 ⎦
```

The same matrix, the same computation, two readings. One decimal, one structural. The determinant of `A` is `3 · 5 − 1 · 2 = 13`, and the chained form makes that visible. Every cell carries `13` as its denominator. The chain works the same way on vectors. Calling `unit.asFractions().asExpression()` on a normalized vector renders the rational form of every component, stacked as a column. See <doc:Determinants-Primer> for the broader treatment of the determinant as a single number that controls invertibility, and see <doc:Fraction> for the type itself.

### When the rendering reveals numerical noise

Most rendered results look the way the math says they should. The decimals match the fractions to four places, the polynomial reads the way it would be written by hand, the matrix lines up cleanly. The cases where the rendering shows something *unexpected* are almost always the rendering being honest about what the underlying `Double` actually holds.

The clearest example comes from `polyfit`. A degree-3 fit to data that is genuinely quadratic does not return a polynomial with three coefficients; it returns one with four, where the leading `x³` coefficient is a residual on the order of `1e-17`. This is not a bug. It is the noise floor of a floating-point least-squares solve on a near-collinear system. The **Vandermonde columns** `x⁰, x¹, x², …` grow more parallel as the degree rises, and the solver returns the best-conditioned answer it can. Without intervention, the rendered polynomial would advertise itself as cubic when it is numerically quadratic. The `relativeZeroTolerance` parameter on `Polynomial.asExpression` is what keeps that honest. Each coefficient is compared against `max(|aⱼ|)` and dropped if its magnitude falls below `tolerance · max(|aⱼ|)`. The default `1e-12` sits two orders above the typical solver noise floor and twelve orders below any coefficient anyone would write by hand:

```swift
import Quiver

// The leading x³ is machine noise; the polynomial is numerically quadratic.
Polynomial([1, 3, 2, 4.3e-17]).asExpression()
// "2x² + 3x + 1"

// Pass 0 to disable suppression and see every coefficient as the solver returned it.
Polynomial([1, 3, 2, 4.3e-17]).asExpression(relativeZeroTolerance: 0)
// "4.3e-17x³ + 2x² + 3x + 1"
```

The tolerance is *relative* rather than absolute by design. A polynomial whose true coefficients all live near `1e-13`, which is rare but possible in physical-constant or small-scale fits, would have every term suppressed by an absolute `1e-12` cutoff. Comparing against `max(|aⱼ|)` makes the test "is this coefficient negligible *relative to the dominant term*," which is the question the reader actually wants answered. See <doc:Numerical-Literacy> for the matching diagnostic at fit time. The `conditionNumber` of a fit's design matrix signals when Vandermonde conditioning has crossed from "noise floor" to "result not reliable."

The rendering preserves the other IEEE-754 signals exactly as Quiver's contract requires. A `Double.nan` cell — a correlation entry on a flat column, the square root of a negative number — renders as `"NaN"`, unmistakable in any context. Positive infinity renders as `"∞"` and negative infinity as `"-∞"`, using the Unicode infinity symbol because it reads unambiguously inside a mathematical expression. Negative zero is normalized to `"0"`. IEEE-754 allows `-0.0` to differ from `+0.0` by sign bit alone, and the normalization keeps a rendered cell from carrying a leading minus that nothing in the math justifies. The rule is consistent across vectors, matrices, polynomials, and standalone scalars: every cell passes through the same formatter.

> Tip: When two rendered values that should be equal differ only in their last digit, the gap is almost always a floating-point artifact rather than a real difference. Compare with a tolerance such as `abs(a - b) < 1e-9` (the same default <doc:Numerical-Literacy> recommends) before treating the disagreement as a bug.

Two boundary behaviors complete the picture. The formatter switches to scientific notation below `|x| < 1e-3` so distinct sub-millisecond values stay distinguishable — `1e-5` renders as `"1e-05"` rather than collapsing to `"0.0000"`. Above that threshold, integer-valued doubles drop the decimal point (`5` not `5.0000`), but a value like `0.99999` keeps its full decimal form (`"1.0000"`) rather than rounding to `"1"`, because advertising a non-integer as an integer would lie about what the `Double` holds. Values right at the `1e-3` boundary can render in either notation — see <doc:Numerical-Literacy> for the broader pattern of why textbook formulas and stable formulations diverge at floating-point boundaries.

The rendered string is not a round-trip-safe encoding. Parsing `"0.3846"` back to a `Double` will not recover the original `5.0 / 13.0` bit-for-bit. Likewise, `Fraction(0.1)` returns `1/10` even though `0.1` is not exactly one-tenth — it is the closest representable binary fraction. The string and the `Fraction` are display views; the `Double` underneath is what the computation uses.

### From reading the answer to trusting it

Rendering is the first half of working with a numeric result, turning the stored representation into a form the reader can interpret. The second half is judging when to trust what the rendering shows. See <doc:Numerical-Literacy> for the broader treatment of floating-point precision, the `nil` versus `NaN` contract, and the cases where the math is well-defined but the conditioning of the inputs is not. See <doc:Determinants-Primer> for the worked example of why the inverse of `[[3, 1], [2, 5]]` shares `13` as a denominator across every cell, and what a determinant near zero says about the matrix's invertibility. See <doc:Polynomials> for the type that the `relativeZeroTolerance` parameter belongs to, including the `polyfit` entry point where machine-noise coefficients first appear.

> Experiment: **The Quiver Notebook** is the right place to see the chained rendering pay for itself. Invert the 3×3 Lo Shu magic square `[[8, 1, 6], [3, 5, 7], [4, 9, 2]]` and chain `.asFractions().asExpression()` on the result. Before running it, compute the determinant by hand. It is `-360`, and that number is what every denominator is built from. Predict that the rendered inverse will carry `360` and its factors as denominators, then run the cell and watch GCD reduction collapse several cells to `90`, `180`, and `45`. The determinant of the original matrix is showing through the inverse, factored down to lowest terms. See <doc:Quiver-Notebook>.
