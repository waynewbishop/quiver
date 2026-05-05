# Quiver Notebook

A browser-based Swift IDE for testing and evaluating Quiver models.

## Overview

The [Quiver Notebook](https://github.com/waynewbishop/quiver-notebook) provides a fast, lightweight environment for learning Quiver and for prototyping. Established as a standalone web-based IDE, it serves two audiences: students who want to learn statistics, linear algebra, and machine learning in Swift, and developers who want a quick iteration loop for testing and building their own models.

### Setting up locally

The Notebook runs on the Swift command-line toolchain and requires macOS 15 (Sequoia) or newer with Swift 5.9 or newer. The lightest way to get Swift on macOS is **swiftly**, Swift's official toolchain installer. It runs as a normal Mac installer and does not require Homebrew or Xcode.

1. Download the installer: [swiftly-1.1.1.pkg](https://download.swift.org/swiftly/darwin/swiftly-1.1.1.pkg)
2. Double-click the downloaded file and follow the prompts.
3. Open a new terminal tab and run:

   ```bash
   ~/.swiftly/bin/swiftly init
   ```

   This downloads the latest Swift toolchain into your home folder and configures your shell.
4. Confirm the install:

   ```bash
   swift --version
   ```

Once Swift is in place, clone and run:

```bash
git clone https://github.com/waynewbishop/quiver-notebook
cd quiver-notebook
swift run
```

Then open `http://localhost:8080` in a browser. The first launch compiles the libraries and the editor, which takes a minute or two on most machines. Every launch after that starts in seconds.

> Important: The local server binds to `127.0.0.1` by design and refuses to start if the address is changed. The Notebook is reachable only from the same machine that launched it — a deliberate constraint, not a configuration bug.

### Writing and running snippets

The editor opens with `Quiver` and `Foundation` already imported, so a working snippet can begin with the first line of real work. Press Cmd+Enter (or Ctrl+Enter on Linux) to compile and run, and output appears in the pane below the editor.

> Tip: Cmd+Enter triggers a full Swift compile of the entire editor contents, not a cell-by-cell evaluation. Compile errors stop the run, and there is no shared state between runs — every snippet is its own complete program.

The Notebook scopes itself deliberately to Quiver — vectors, matrices, statistics, and models — so students see the same surface every time they open the editor. There is no plugin system and no extra package configuration to learn.

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

The `guard let` checks that the dataset loaded successfully. If the file is missing or unreadable, the `else` branch runs and we call `exit(0)` to end the program cleanly. (Inside an Xcode project the same check would `return` from a function — in the Notebook there is no enclosing function, so `exit(0)` is the equivalent.)

Once we have a `Panel`, the rest of Quiver is one method call away. We can ask for descriptive statistics, split the data into training and test sets, or pull selected columns into a matrix for a model to learn from. See <doc:Panel> for the full surface and <doc:Train-Test-Split> for the partitioning method.

> Experiment: Open any of the bundled examples and edit it. The editor auto-saves to the browser as we type, so refreshing the page or closing and reopening the tab preserves work in progress.

### Carrying snippets into an app

A model fitted in the Notebook is an ordinary Swift value, and Quiver's models are `Codable` — a trained model can be saved to disk from the Notebook and read back inside an iOS or watchOS app:

```swift
let model = try LinearRegression.fit(features: features, targets: targets)
let data = try JSONEncoder().encode(model)
try data.write(to: modelURL)
```

The decoded model behaves identically to the one that was trained — same coefficients, same predictions. See <doc:Model-Persistence> for the full save-and-load workflow, and <doc:Concurrency-Primer> for sharing a decoded model across concurrent code.

### Notebook Datasets

The Notebook ships with a small library of bundled teaching datasets — iris measurements, housing prices, exam scores, and other classics that make it possible to write a working snippet without first hunting for a CSV. Each dataset loads through a single `Dataset.<name>` call and exposes a `toPanel()` method that hands back a named-column table ready for descriptive statistics, train-test splits, or a fitted model. For the full library and the recipes built on top of it, see <doc:Notebook-Datasets>.

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
