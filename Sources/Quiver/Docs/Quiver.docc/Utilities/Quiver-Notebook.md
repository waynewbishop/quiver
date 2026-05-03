# Quiver Notebook

Writing and running Swift in a web browser, with Quiver ready to use.

## Overview

The Quiver Notebook is the fastest way to try a Quiver idea — a vector calculation, a statistical check, a model fit without setting up a project. Clone a repository and a browser tab opens with `Quiver`, `Foundation`, and a small library of teaching datasets ready to use. Nothing else needs to be installed beyond the Swift CLI.

> Tip: The Notebook lives in its own repository on GitHub. This page covers how to start it, how snippets fit alongside the rest of Quiver, and what the Notebook adds beyond a plain Swift project. For the bundled teaching datasets, see <doc:Notebook-Datasets>.

### Setting up locally

The Notebook runs on the Swift command-line toolchain, which is not preinstalled on macOS. Confirm Swift 5.9 or newer is available before cloning:

```bash
swift --version
```

If the command is missing or older than 5.9, the toolchain is a free download from [swift.org/install](https://www.swift.org/install/). Xcode is not required — the standalone toolchain is enough. Once Swift is in place, we clone and run:

```bash
git clone https://github.com/waynewbishop/quiver-notebook
cd quiver-notebook
swift run
```

Then open `http://localhost:8080` in a browser. The first launch compiles the libraries and the editor, which takes a minute or two on most machines. Every launch after that starts in seconds.

> Important: The local server binds to `127.0.0.1` by design and refuses to start if the address is changed. The Notebook is reachable only from the same machine that launched it — a deliberate constraint, not a configuration bug.

### Writing and running snippets

The editor opens with `Quiver`, `Structures`, and `Foundation` already imported, so a working snippet can begin with the first line of real work. Press Cmd+Enter (or Ctrl+Enter on Linux) to compile and run, and output appears in the pane below the editor. The editor auto-saves to the browser's local storage, so refreshing the page does not lose code.

> Tip: Cmd+Enter triggers a full Swift compile of the entire editor contents, not a cell-by-cell evaluation. Compile errors stop the run, and there is no shared state between runs — every snippet is its own complete program.

Two libraries cover the ground a typical lesson needs. Quiver provides the numerical and machine-learning surface — vectors, matrices, statistics, and models. Structures provides the classic data structures that appear in algorithms courses: heaps, tries, graphs, stacks, queues, and binary search trees. The Notebook deliberately scopes itself to these two packages, so students see the same surface every time they open it.

### A first snippet

The first snippet most students write loads a bundled dataset, pulls out the table of values, and prints a few rows. The same three steps — load, extract, inspect — apply to every dataset in the library:

```swift
guard let iris = Dataset.iris else {
    exit(0)
}

let panel = iris.toPanel()
print(panel.head(n: 3))
print(iris.description)
print("shape:", panel.shape)
```

A short note on a Swift detail visible in this snippet: `guard let` checks that the dataset loaded successfully. If the file is missing or unreadable, the `else` branch runs and we call `exit(0)` to end the program cleanly. (Inside an Xcode project the same check would `return` from a function — in the Notebook there is no enclosing function, so `exit(0)` is the equivalent.)

Once we have a `Panel`, the rest of Quiver is one method call away. We can ask for descriptive statistics, split the data into training and test sets, or pull selected columns into a matrix for a model to learn from. See <doc:Panel> for the full surface and <doc:Train-Test-Split> for the partitioning method.

### Carrying snippets into an app

A model fitted in the Notebook is an ordinary Swift value, and Quiver's models are `Codable` — a trained model can be saved to disk from the Notebook and read back inside an iOS or watchOS app:

```swift
let model = try LinearRegression.fit(features: features, targets: targets)
let data = try JSONEncoder().encode(model)
try data.write(to: modelURL)
```

The decoded model behaves identically to the one that was trained — same coefficients, same predictions. See <doc:Model-Persistence> for the full save-and-load workflow, and <doc:Concurrency-Primer> for sharing a decoded model across concurrent code.

### Pinned releases

Each clone of the Notebook is locked to a specific Quiver release, so a Tuesday lecture and a Thursday exam will run against identical code. The active version is displayed in the footer of the editor, which makes it easy for a class to confirm everyone is on the same release. When a new Quiver version is bundled into the Notebook, it lands on the upstream `main` branch — we pull when we are ready to move forward, or stay on the version a course or workshop started with.

### Privacy and network behavior

Everything stays on the machine where it was written. The Notebook does not create accounts, send telemetry, or contact analytics endpoints, and the local server only accepts connections from the same machine — the bind address is fixed at `127.0.0.1` and the server refuses to start if it is changed. Bundled datasets ship with the repository and are read from disk by the local process. CSVs loaded from a custom path are also read locally and never transmitted. For classroom labs, exam settings, and air-gapped environments, the Notebook is designed to be acceptable on a school's standard machine image without IT review.

> Tip: Forking the Notebook for a course, adding custom examples, pinning a specific `Quiver` release for a semester, and switching ports are covered in <doc:Quiver-Notebook-For-Classrooms>.

### Related
- <doc:Notebook-Datasets>
- <doc:Quiver-Notebook-For-Classrooms>
- <doc:Panel>
- <doc:Model-Persistence>
- <doc:Train-Test-Split>
