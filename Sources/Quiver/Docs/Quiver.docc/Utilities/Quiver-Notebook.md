# Quiver Notebook

A browser-based Swift IDE for testing and evaluating Quiver models.

## Overview

The [Quiver Notebook](https://github.com/waynewbishop/quiver-notebook) provides a fast, lightweight environment for learning Quiver and for prototyping. Established as a standalone web-based IDE, it serves two audiences: students who want to learn statistics, linear algebra, and machine learning in Swift, and developers who want a quick iteration loop for testing and building their own models.

### Setting up locally

The Notebook runs on the Swift command-line toolchain and requires macOS 15 (Sequoia) or newer with Swift 6.0 or newer.

#### 1. Install Homebrew

> Note: Homebrew is the standard **package manager** for macOS — a free tool that installs developer software from a single command line. We recommend Homebrew because it downloads the Swift toolchain installer, keeps it updated alongside other developer tools, and makes the whole install reproducible. Homebrew installs can also be uninstalled cleanly with a single command.

Open the **Terminal** application (in Applications → Utilities, or search "Terminal" in Spotlight). Paste this command and press Return — it downloads and runs the official Homebrew install script from the [Homebrew](https://brew.sh) website:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

When the install finishes, confirm Homebrew is working by checking its version:

```bash
brew --version
```

The terminal should print a line like `Homebrew 4.4.0`. If it says "command not found," the install didn't add Homebrew to the terminal's search path — close the terminal window, open a new one, and try again.

#### 2. Install Swiftly through Homebrew

**Swiftly** is Swift's official toolchain installer, maintained by the Swift project itself. Homebrew is a general-purpose package manager for macOS — we are using it here to install Swiftly. Swiftly is what installs the actual Swift compiler.

Install Swiftly:

```bash
brew install swiftly
```

Confirm Homebrew is managing the install:

```bash
brew list swiftly
```

This prints the list of files Homebrew installed for the Swiftly package. A successful install shows several file paths. An empty result or an error means the install didn't complete.

#### 3. Install the Swift toolchain

With swiftly installed, run it once to download and configure the Swift compiler itself:

```bash
swiftly init
```

The download takes a few minutes. When it finishes, confirm Swift is available:

```bash
swift --version
```

The terminal should print a Swift version like `Swift version 6.0.2`. If it says "command not found," close the terminal window and open a new one — the install only takes effect in new terminal sessions.

#### 4. Clone and run the Notebook

The Notebook lives in its own GitHub repository. Clone a copy to the Mac and start it:

```bash
git clone https://github.com/waynewbishop/quiver-notebook
cd quiver-notebook
swift run
```

The first launch takes a minute or two while Swift compiles the Notebook and pre-warms the snippet sandbox. The sandbox is a separate Swift package that depends on Quiver, so the first launch downloads and compiles Quiver from GitHub. Subsequent runs use the cached build and start in seconds. When the server is ready, the terminal prints a banner:

```
Quiver Notebook is running.

Open this URL in your browser:
http://localhost:8080

If port 8080 is in use, restart with: PORT=8090 swift run
```

Open the URL in any browser to start writing snippets. Press `Ctrl+C` in the terminal to stop the server.

> Important: The local server binds to `127.0.0.1` by design and refuses to start if the address is changed. The Notebook is reachable only from the same machine that launched it — a deliberate constraint.

### Writing and running snippets

The editor opens with `Quiver` and `Foundation` already imported, so a working snippet can begin with the first line of real work. Press Cmd+Enter (or Ctrl+Enter on Linux) to compile and run, and output appears in the pane below the editor. The editor auto-saves to the browser's local storage, so refreshing the page does not lose code.

Hovering over a Quiver symbol shows its signature and a short description pulled directly from Quiver's documentation. Option-clicking (or Alt-clicking) a symbol opens a larger reference popup with the full doc comment. The same documentation that lives in the DocC catalog is available inline as snippets are written.

The font size has four presets: Tiny, Normal, Large, and Presenter. **Presenter** mode sizes the editor text for projection during a lecture or workshop, so code stays readable from the back of a room without resizing the browser window.

> Note: Cmd+Enter triggers a full Swift compile of the entire editor contents, not a cell-by-cell evaluation. Compile errors stop the run, and there is no shared state between runs — every snippet is its own complete program.

Quiver provides the numerical and machine-learning surface — vectors, matrices, statistics, and models. The Notebook deliberately scopes itself to this one package, so students see the same surface every time they open it.

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

Code and data stay on the machine where they were written. The Notebook does not create accounts, send telemetry, or contact analytics endpoints, and the local server only accepts connections from the same machine — the bind address is fixed at `127.0.0.1` and the server refuses to start if it is changed. Bundled datasets ship with the repository and are read from disk by the local process. CSVs loaded from a custom path are also read locally and never transmitted.

The editor itself uses Monaco — Microsoft's open-source code editor — which the browser loads from a public CDN (`cdnjs.cloudflare.com`) on each page load. This is a one-way asset fetch with no code or data sent back. Schools that block CDN access or require fully offline environments should plan to either allow `cdnjs.cloudflare.com` or vendor Monaco locally before adoption.

> Tip: Forking the Notebook for a course, adding custom examples, pinning a specific `Quiver` release for a semester, and switching ports are covered in <doc:Quiver-Notebook-For-Classrooms>.

### Troubleshooting

A few setup steps can fail in predictable ways. Each item below describes the symptom, the cause, and the fix.

#### Homebrew command not found

After installing Homebrew, the terminal reports `brew: command not found` when running `brew --version`. Homebrew finished installing but didn't add itself to the terminal's search path. Close the terminal window and open a new one, then try `brew --version` again. If a fresh terminal still doesn't recognize `brew`, re-read the install output — Homebrew prints a "Next steps" message at the end with the exact command to run.

#### Swift command not found after init

After running `swiftly init`, the terminal reports `swift: command not found` when running `swift --version`. Close the terminal window and open a new one. The install adds Swift to the shell, but the change only applies to newly opened terminals.

If a fresh terminal still doesn't recognize `swift`, run `swiftly init` once more. It's safe to run again and will finish the setup.

#### Port already in use

Something else on the machine is already serving on port 8080 — typically another Notebook session that didn't shut down cleanly, or an unrelated development server. Quit the other process, or restart the Mac if it's unclear what's holding the port. The Notebook can also be moved to a different port — see <doc:Quiver-Notebook-For-Classrooms>.

#### First launch hangs while resolving dependencies

The first launch needs internet access to fetch Quiver from GitHub for the snippet sandbox. On a slow or restricted network the download can take several minutes. Wait a few minutes before assuming something is wrong.

If the build never proceeds past `Resolving dependencies`, check the network connection. A captive-portal page (common on hotel and campus wifi) blocks SwiftPM silently — open a browser, complete the wifi sign-in, then re-run `swift run`.

#### Browser cannot reach the Notebook

The Notebook serves only on the local machine and only on `http://localhost:8080`. Confirm the terminal still shows the `Quiver Notebook is running.` banner — if it doesn't, the server has stopped. Run `swift run` again.

#### A previously saved snippet is missing

The editor auto-saves to the browser's local storage, which is scoped to a single browser on a single machine. Snippets do not persist if the browser cache is cleared, if a different browser is used, or if the Notebook is opened on another computer.

> Tip: For coursework that needs to survive between sessions, copy snippets into a separate file before closing the browser.
