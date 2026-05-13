# Exploring Quiver with Xcode

Using the playground macro to inspect Quiver values inside an existing project.

## Overview

The `#Playground` macro, introduced in [Xcode 26](https://developer.apple.com/xcode/), turns any Swift file inside a project into an interactive surface — write an expression, see the result inline in the Canvas, no build-and-run cycle. For a project that already depends on Quiver, this is the most direct way to inspect a value, verify a calculation, or sanity-check a method without leaving the codebase.

> Important: The `#Playground` macro is not the same as a `.playground` file. Traditional `.playground` files run in an isolated sandbox and cannot import Swift packages. The `#Playground` macro compiles as part of the project, so it has full access to SPM dependencies, including Quiver, with no extra configuration. The distinction trips up nearly every first-time user.

### Writing a playground inside a project

Add a new Swift file to the project, import `Playgrounds`, and wrap the code in a `#Playground` block. The block compiles and runs as part of the project, so it can use any Quiver type the project already has access to:

```swift
import Playgrounds
import Quiver

#Playground {
    let v = [3.0, 4.0]
    let length = v.magnitude              // 5.0
    let unit = v.normalized               // [0.6, 0.8]
    let fractions = unit.asFractions()    // [3/5, 4/5]
    print("length: \(length), unit: \(fractions)")
}
```

The Canvas shows the result inline as the code is written, and re-evaluates on every edit. Changing the input vector, swapping a method, or adjusting a parameter updates the result without a restart.

### Naming blocks for parallel experiments

Name `#Playground` blocks to organize separate inspections in one file. Each named block runs independently, so a single file can hold a small suite of side-by-side comparisons:

```swift
import Playgrounds
import Quiver

#Playground("Dot product") {
    let a = [1.0, 2.0, 3.0]
    let b = [4.0, 5.0, 6.0]
    print(a.dot(b))                // 32.0
}

#Playground("Cosine similarity") {
    let a = [1.0, 2.0, 3.0]
    let b = [4.0, 5.0, 6.0]
    print(a.cosineOfAngle(with: b))  // 0.97
}
```

Named blocks are useful for comparing related operations side by side, or for working through a series of inspections in a single file without losing track of what each one shows.

### When to reach for the Notebook

The `#Playground` macro is the right tool when Quiver is already a dependency of a project under active development. For exploration outside an existing project — running a snippet during a lecture, working through the cookbook, or letting students share a uniform environment — the <doc:Quiver-Notebook> is the better surface. The Notebook ships with Quiver pre-imported, requires no project setup, and runs from a clone-and-run repository. The two surfaces are complementary: `#Playground` for in-project inspection, Notebook for everything else.

> Experiment: [quiver-cookbook](https://github.com/waynewbishop/quiver-cookbook) is built entirely on the `#Playground` macro. Cloning the repo and opening any single recipe — wind-tunnel lift prediction, semantic search, sensor-driven driving decisions — drops a working Quiver example into the Canvas with no setup. Watching a recipe evaluate inline is the fastest way to see what the macro feels like at full scale.

### Related
- <doc:Quiver-Cookbook>
- <doc:Quiver-Notebook>
- <doc:Notebook-Datasets>

