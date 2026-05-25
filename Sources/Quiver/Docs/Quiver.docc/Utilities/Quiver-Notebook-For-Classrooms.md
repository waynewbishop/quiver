# Quiver Notebook for Classrooms

Adopting the Notebook in a course, workshop, or self-study setting.

## Overview

The Quiver Notebook gives a class a Swift environment every student can run on their own laptop with one command — no accounts, no shared server, no IT review, and no per-student setup. An instructor distributes the environment by sharing a repository URL, and each student clones it, runs `swift run`, and arrives at the same editor as everyone else. This page covers the configuration choices that come up when adopting the Notebook in a class, a workshop, or a self-study setting.

> Note: For starting the Notebook and writing snippets, see <doc:Quiver-Notebook>.

### When to reach for the Notebook

The Notebook is the right tool for:

- **Teaching a Swift-based ML course** — a pure-Swift classroom environment that runs on every student's machine.
- **Running exercises locally** — classroom labs and self-study where each student runs the Notebook on their own laptop.
- **Student self-study** — anyone reading *Swift Algorithms & Data Structures* who wants to experiment alongside the book.
- **Prototyping ML for Apple devices** — designing a model in pure Swift before dropping it into an iOS, watchOS, or visionOS app.
- **Workshops and tutorials** — a shared environment attendees can clone, run, and keep after the session.

### Who the Notebook is for

**Students** working through a course, textbook, or self-study get a Swift environment that does numerical work without installing additional binaries, configuring system packages, or downloading datasets separately. One clone and one command produce a working editor with Quiver, Foundation, and the bundled datasets already wired in.

**Educators** preparing lectures or assignments can fork the repository, drop custom example files into the `examples-custom/` folder, and distribute the URL to a class. The bundled stack covers enough ground for an applied linear algebra unit, an introductory descriptive statistics segment, and an applied regression module — material that fits inside an existing course. A short supervised learning survey covering k-nearest neighbors, k-means, and Naive Bayes also fits comfortably in a few weeks.

**Apple-platform developers** prototyping a model or testing an idea get a focused editor without a project workspace. Code that runs here compiles unchanged on any Apple platform — including iOS, watchOS, visionOS, and Vapor server-side targets.

### Adding your own examples

The Notebook's left sidebar is populated by reading every `.swift` file in two directories: `examples/` for the bundled snippets that ship with the repository, and `examples-custom/` for files added by an instructor. An instructor extends the list by dropping new files into `examples-custom/` — there is no plugin system, no manifest, and no rebuild required. Restart `swift run` to pick up newly added files. Custom entries appear in the sidebar after the bundled set, using the title from each file's `// Title:` comment.

Each example file begins with a `// Title:` comment on the first line, and the text after the colon becomes the sidebar label that students see:

```swift
// Title: Class Quiz Scores
//
// Print descriptive statistics for a small sample of quiz scores.

let scores = [82.0, 91.0, 76.0, 88.0, 95.0, 73.0, 84.0, 90.0]
print("mean:    ", scores.mean() ?? 0)
print("std dev: ", scores.standardDeviation() ?? 0)
print("median:  ", scores.median() ?? 0)
```

The bundled examples are numbered (`01-…`, `02-…`) so they appear in the order a course would introduce them. Custom assignments can use the same numbering convention to slot into the sequence, or use a different prefix (`hw1-…`, `lab3-…`) to keep coursework visually separate from the built-in set.

### Distributing the Notebook to a class

The recommended distribution model is fork-and-clone. A *fork* is a personal copy of a GitHub repository under your own account; on github.com, click the **Fork** button in the upper right of the `quiver-notebook` page. An instructor forks the `quiver-notebook` repository to a course-specific copy, adds custom examples, optionally locks down a particular Quiver release, and shares the fork's URL with students. Each student clones once, runs `swift run`, and arrives at the same environment as everyone else in the class.

A typical fork preparation looks like this:

```bash
git clone https://github.com/your-org/cs180-quiver-notebook
cd cs180-quiver-notebook

cp ~/lectures/week3-pca.swift examples-custom/30-week3-pca.swift
cp ~/lectures/week4-knn.swift examples-custom/40-week4-knn.swift

git add examples-custom/
git commit -m "Week 3 and 4 lecture examples"
git push
```

Students then clone the fork directly:

```bash
git clone https://github.com/your-org/cs180-quiver-notebook
cd cs180-quiver-notebook
swift run
```

The first launch takes a minute or two while Swift compiles the Notebook and pre-warms the snippet sandbox. The sandbox depends on Quiver, so the first launch downloads Quiver from GitHub. Subsequent runs start in seconds.

### Pinning a version for a semester

A pinned version means every student in the class is running identical code — the same APIs, the same behavior, the same examples — for the duration of a course. The Notebook ships with a specific Quiver release locked in by its package manifest, so every clone resolves to the same known-good version. The footer of the editor displays the active version, which makes it easy to confirm before an exam or graded assignment.

To pin Quiver to a specific version for the duration of a course, edit the `Package.swift` in the Notebook repository and change the Quiver dependency from a range to an exact version:

```swift
.package(url: "https://github.com/waynewbishop/quiver", .exact("1.1.0"))
```

After saving the file, run `swift package resolve` to record the pinned version in `Package.resolved`. Every student who clones the repository will then build against the same Quiver version, regardless of what gets released during the semester.

To hold a course on a specific release across a semester, do not pull from the Notebook's `main` branch until the course is over. New Quiver releases are bundled and pushed to `main` on the upstream repository — staying on the fork's locked manifest keeps the environment frozen until the instructor is ready to move forward.

> Note: `swift package update` resolves dependencies to newer versions that satisfy the manifest's version requirements. For range requirements like `from: "1.1.0"`, this can move to any 1.x release. For exact requirements like `.exact("1.1.0")`, the version is fixed and `swift package update` has no effect. The regular `swift run` and `swift build` commands respect whichever requirement is in the manifest.

### Privacy and network behavior

Code and data stay on the student's machine, the local server accepts connections only from that machine, and there are no accounts, telemetry, or analytics anywhere in the stack. The one outbound request is the browser fetching the Monaco editor from a public CDN at page load, which schools that restrict CDN access can allow or vendor locally. See <doc:Quiver-Notebook> for the full account of the Notebook's privacy and network behavior.

> Tip: The Notebook is designed for one student per laptop and runs Swift with the permissions of whoever launched it. A shared classroom server is not a supported configuration — a multi-user deployment would need sandboxing, resource limits, and per-user isolation that the current scope does not include.

### Changing the default port

The Notebook listens on port `8080` by default. If another process is already using that port — common on machines that also run a local development server — we set the `PORT` environment variable before launching:

```bash
PORT=8090 swift run
```

The active port is logged to the terminal at startup, and the browser tab needs to point at the same one (`http://localhost:8090` for the example above). Any unused TCP port works.

> Tip: Changing the port does not change the bind address. The server still listens only on `127.0.0.1` and refuses connections from elsewhere on the network — a custom port is a convenience choice, not a way to expose the Notebook to other machines.

### Troubleshooting

#### Merge conflict after upgrading the Notebook

When custom example files are committed into the fork, pulling future upstream Notebook updates may produce a merge conflict on `examples-custom/README.md` if the upstream README has changed. This affects only the README — the custom `.swift` files in `examples-custom/` are never modified by upstream changes, so the conflict is limited to one file. Resolve it by keeping the upstream version of the README (the only file the upstream Notebook owns inside `examples-custom/`), then continue the pull.

### Carrying student work into apps

The most direct payoff for a course aimed at Apple-platform engineers: students prototype in the Notebook, then carry the same code into an Xcode project for the app-development unit. There is no separate notebook language and no hidden cell behavior to translate away — the only difference between a snippet here and the same code in an app is that the app writes its own `import Quiver` at the top of each file, where the Notebook supplies that line for us.

For saving a trained model from the Notebook and loading it inside an app, see <doc:Model-Persistence>. For the model APIs themselves, see <doc:Linear-Regression>, <doc:Naive-Bayes>, <doc:Nearest-Neighbors-Classification>, and <doc:KMeans-Clustering>.
