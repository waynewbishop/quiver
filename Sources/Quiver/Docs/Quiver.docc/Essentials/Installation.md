# Installation

@Metadata {
  @TitleHeading("Getting Started")
}

Add Quiver to an Xcode project or Swift package.

## Overview

Quiver is distributed as a Swift package with zero external dependencies. It supports iOS 15+, macOS 12+, tvOS 15+, watchOS 8+, and visionOS 1+, and requires Swift 5.9 or later. It also runs in server-side Swift environments like Vapor, Linux, and containerized deployments.

### Adding Quiver to an Xcode project

Open a project in Xcode and navigate to **File → Add Package Dependencies**. In the search field, enter the repository URL:

```
https://github.com/waynewbishop/quiver
```

Set the dependency rule to **Up to Next Major Version** starting from `1.0.0`, then click **Add Package**. Xcode resolves and downloads the package automatically.

### Adding Quiver to a Swift package

Add Quiver as a dependency in `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/waynewbishop/quiver", from: "1.0.0")
]
```

Then add it to the target that needs it:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "Quiver", package: "quiver")
    ]
)
```

### Verifying the installation

Import Quiver at the top of any Swift file and run a quick check:

```swift
import Quiver

let a: [Double] = [1, 2, 3]
let b: [Double] = [4, 5, 6]

// compute the dot product of two vectors
let result = a.dot(b)
print(result) // 32.0
```

> Tip: For learning Quiver and prototyping models against bundled teaching datasets, the <doc:Quiver-Notebook> runs Swift snippets in a browser tab with `Quiver` and `Foundation` already imported. Best for students, instructors, and developers building models before dropping them into an app.
