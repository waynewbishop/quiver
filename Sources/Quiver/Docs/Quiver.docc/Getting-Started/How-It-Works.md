# How It Works

Understanding Quiver's architecture and how it works with Swift.

## Overview

Quiver adds numerical computing capabilities directly to Swift's standard `Array` type using a language feature called **extensions**. Rather than introducing custom container types, Quiver works with native Swift `Array`. The result is that a plain `[Double]` gains mathematical operations without becoming a different type.

```swift
import Quiver

// This is a standard Swift Array<Double> — nothing special
let position = [3.0, 4.0]

// Quiver adds .magnitude directly to the array
position.magnitude  // 5.0
```

In this example, the `position` array could be passed to SwiftUI, Swift Charts, SwiftData, or any other Swift API.

### What are extensions

An **extension** in Swift adds new methods, computed properties, and initializers to a type that already exists — even one we did not write. The capability is added directly to the type, and every instance gains it automatically.

> Tip: To learn more about Swift's type system, generics, and protocol-oriented architecture see [Swift Algorithms & Data Structures](https://waynewbishop.github.io/swift-algorithms/) — the companion book to this framework.

Here is a simple extension that adds a `doubled` property to `Int`:

```swift
extension Int {
    // Every Int in the project now has this property
    var doubled: Int { self * 2 }
}

let value = 5
value.doubled  // 10
```

Swift's standard library uses this same mechanism extensively. Methods like `.sorted()`, `.reversed()`, and `.contains()` are all added to `Array` through extensions — they are not part of the core array definition. Quiver follows the same pattern to add numerical operations.

> Tip: Extensions can add new functionality to a type, but they cannot override or modify existing behavior. Quiver's extensions are purely additive — they never change how `Array` already works.

### Constrained extensions

Not every array operation makes sense for every element type. Computing a mean requires division, which integers cannot do precisely. Computing magnitude requires a square root, which strings cannot produce. Quiver needs a way to say "this method only exists when the elements are the right kind of number."

Swift solves this with **constrained extensions** — an extension that applies only when the generic type parameter meets a specific requirement. The constraint acts as a gate: if the elements qualify, the methods appear. If they do not, the compiler prevents us from calling them:

```swift
//Actual Quiver extension that adds `.magnitude` to arrays
extension Array where Element: FloatingPoint {
    var magnitude: Element {
        var sumOfSquares = Element.zero
        for element in self {
            sumOfSquares += element * element
        }
        return sumOfSquares.squareRoot()
    }
}
```

The `where Element: FloatingPoint` clause is the gate. It means `.magnitude` appears on `[Double]` and `[Float]` but not on `[String]`, `[Int]`, or any other array type. The compiler enforces this at build time — not with a runtime error, but by making the method invisible to types that do not qualify.

This is the Pythagorean theorem expressed in Swift: sum the squares of each element, then take the square root. For the vector `[3.0, 4.0]`, the calculation is `√(3² + 4²) = √(9 + 16) = √25 = 5.0`.

### The type system at work

In Swift, the compiler verifies that every operation we call is mathematically valid for the element type we are using. If it is not, the code does not build.

Consider what happens when we try to call operations on the wrong array type:

```swift
let integers = [1, 2, 3, 4, 5]
let doubles = [3.0, 4.0, 5.0]

// These compile — the constraints are satisfied
integers.sum()           // Numeric ✓
doubles.mean()           // FloatingPoint ✓

// These do NOT compile — the compiler catches the error
integers.mean()          // Int is not FloatingPoint
integers.magnitude       // Int is not FloatingPoint
```

Each failed call is caught before the code ever runs. The compiler tells us exactly which protocol the element type is missing. When we refactor an array from `[Double]` to `[Int]`, the build itself flags every Quiver call that no longer applies — `mean()`, `magnitude`, `normalized`, `dot()`. We do not discover these issues through test failures or runtime crashes.

#### Dimensions encoded in types

Swift's type system also encodes dimensionality. The compiler distinguishes between `[Double]` (a vector) and `[[Double]]` (a matrix) at compile time, so there is no need for a runtime property to query how many dimensions an array has — the type signature already tells us. Quiver's `.shape` property is constrained to nested arrays, so calling it on a flat `[Double]` is a compile-time error, not a runtime surprise.

> Tip: For a detailed look at `.shape`, `.size`, and working with matrix dimensions, see <doc:Shape-And-Size>.

#### Named tuples as return types

When Quiver returns structured data, Swift's **named tuples** make each value self-documenting:

```swift
import Quiver

let sales: [[Double]] = [
    [120, 95, 110, 130, 140, 200, 180],
    [85, 70, 90, 75, 95, 150, 130]
]

// .shape returns (rows: Int, columns: Int)
let dimensions = sales.shape
dimensions.rows     // 2
dimensions.columns  // 7
```

The labels `.rows` and `.columns` are built into the return type. We cannot accidentally swap them, and every call site reads like documentation. The same pattern appears throughout the framework. The `.quartiles()` method returns a named tuple with `.min`, `.q1`, `.median`, `.q3`, `.max`, and `.iqr` — six values, each accessible by name:

```swift
let scores = [72.0, 85.0, 91.0, 68.0, 95.0, 88.0, 76.0]
if let q = scores.quartiles() {
    q.median  // 85.0
    q.iqr     // 15.5
}
```

#### Tuple destructuring

Because these return types are tuples, Swift lets us unpack them directly into named constants:

```swift
let (stores, days) = sales.shape
// stores == 2, days == 7
```

This is **tuple destructuring** — we choose names that match our domain rather than the generic `.rows` and `.columns`. The binding works by position, so the first value is always the row count and the second is always the column count. The result is code that reads like a sentence: "this data has two stores and seven days."

### Comparing results

Quiver's models support direct comparison with `==`. This makes it straightforward to verify that two training runs produce the same clusters, confirm that a confusion matrix matches expectations, or check whether feature scaling preserved the original configuration:

```swift
import Quiver

let run1 = KMeans.fit(data: points, k: 3, seed: 42)
let run2 = KMeans.fit(data: points, k: 3, seed: 42)

// Same seed, same data — same model
run1 == run2  // true
```

No need to serialize, compare properties one at a time, or write custom comparison logic. This works across all four models — `KMeans`, `LinearRegression`, `KNearestNeighbors`, and `GaussianNaiveBayes` — as well as result types like `Cluster`, `Classification`, `ConfusionMatrix`, and `FeatureScaler`. Unit tests can use `XCTAssertEqual` directly.

### A focused, intentional scope

Quiver is designed for educational use, on-device ML features, and data science workflows where understanding the mathematics matters as much as the result. GPU acceleration, automatic differentiation, and distributed training are outside that scope as they bring significant complexity: external dependencies, platform restrictions, and a steeper learning curve that would work against the framework's core goals of clarity, portability, and zero-dependency deployment.

### Performance characteristics

The design prioritizes clarity and portability — the same code runs identically on macOS, iOS, watchOS, visionOS, and Linux. Most operations — vector arithmetic, statistics, broadcasting, boolean masking, element-wise math — are linear and scale predictably to millions of elements.

> Important: The operations worth thinking about are matrix multiplication, matrix inversion, and pairwise comparisons like `findDuplicates(threshold:)` and `clusterCohesion()`. These grow quadratically or cubically with input size, so they perform well for the hundreds-to-low-thousands range typical in educational and on-device use cases.
