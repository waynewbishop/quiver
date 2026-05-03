# Quiver Notebook for Classrooms

Adopting the Notebook in a course, workshop, or self-study setting.

## Overview

The Quiver Notebook gives a class a Swift environment every student can run on their own laptop with one command — no accounts, no shared server, no IT review, and no per-student setup. An instructor distributes the environment by sharing a repository URL, and each student clones it, runs `swift run`, and arrives at the same editor as everyone else. This page covers the configuration choices that come up when adopting the Notebook in a class, a workshop, or a self-study setting.

For starting the Notebook and writing snippets, see <doc:Quiver-Notebook>. For the bundled dataset library, see <doc:Notebook-Datasets>.

### When to reach for the Notebook

The Notebook is the right tool for:

- **Teaching a Swift-based ML course** — a pure-Swift classroom environment that runs on every student's machine.
- **Running exercises in restricted networks** — classroom labs, exam settings, air-gapped environments.
- **Student self-study** — anyone reading *Swift Algorithms & Data Structures* who wants to experiment alongside the book.
- **Prototyping ML for Apple devices** — designing a model in pure Swift before dropping it into an iOS, watchOS, or visionOS app.
- **Workshops and tutorials** — a shared environment attendees can clone, run, and keep after the session.

### Who the Notebook is for

**Educators** preparing lectures or assignments can fork the repository, drop custom example files into the `examples/` folder, and distribute the URL to a class. The bundled stack covers enough ground for an applied linear algebra unit, an introductory descriptive statistics segment, and an applied regression module — material that fits inside an existing course rather than replacing one. A short supervised learning survey covering k-nearest neighbors, k-means, and Naive Bayes also fits comfortably in a few weeks.

**Students** working through a course, textbook, or self-study get a Swift environment that does numerical work without installing additional binaries, configuring system packages, or downloading datasets separately. One clone and one command produce a working editor with Quiver, Structures, Foundation, and the bundled datasets already wired in.

**iOS and Apple-platform developers** prototyping a model or testing an idea get a focused editor without a project workspace. Code that runs here compiles unchanged on any Apple platform — including iOS, watchOS, visionOS, and Vapor server-side targets.

### Adding your own examples

The Notebook's left sidebar is populated by reading every `.swift` file in the `examples/` directory at startup. An instructor extends the list by dropping new files into that folder — there is no plugin system, no manifest, and no rebuild required. Refresh the browser tab and the new entries appear.

Each example file begins with a `// Title:` comment on the first line, and the text after the colon becomes the sidebar label that students see:

```swift
// Title: Class Quiz Scores
//
// Print descriptive statistics for a small sample of quiz scores.

let scores = [82.0, 91.0, 76.0, 88.0, 95.0, 73.0, 84.0, 90.0]
print("mean:    ", scores.mean)
print("std dev: ", scores.standardDeviation)
print("median:  ", scores.median)
```

The bundled examples are numbered (`01-…`, `02-…`) so they appear in the order a course would introduce them. Custom assignments can use the same numbering convention to slot into the sequence, or use a different prefix (`hw1-…`, `lab3-…`) to keep coursework visually separate from the built-in set.

### Distributing the Notebook to a class

The recommended distribution model is fork-and-clone. An instructor forks the `quiver-notebook` repository to a course-specific copy, adds custom examples, optionally locks down a particular Quiver release, and shares the fork's URL with students. Each student clones once, runs `swift run`, and arrives at the same environment as everyone else in the class.

A typical fork preparation looks like this:

```bash
git clone https://github.com/your-org/cs180-quiver-notebook
cd cs180-quiver-notebook

cp ~/lectures/week3-pca.swift examples/30-week3-pca.swift
cp ~/lectures/week4-knn.swift examples/40-week4-knn.swift

git add examples/
git commit -m "Week 3 and 4 lecture examples"
git push
```

Students then clone the fork directly:

```bash
git clone https://github.com/your-org/cs180-quiver-notebook
cd cs180-quiver-notebook
swift run
```

The first launch takes a minute or two while Swift fetches Quiver, Structures, and the editor framework. Subsequent runs start in seconds, and a class running in an air-gapped lab can clone-and-build once on a connected machine, then redistribute the fully built directory.

### Pinning a version for a semester

A pinned version means every student in the class is running identical code — the same APIs, the same behavior, the same examples — for the duration of a course. The Notebook ships with a specific Quiver release locked in by its package manifest, so every clone resolves to the same known-good version. The footer of the editor displays the active version, which makes it easy to confirm before an exam or graded assignment.

To hold a course on a specific release across a semester, do not pull from the Notebook's `main` branch until the course is over. New Quiver releases are bundled and pushed to `main` on the upstream repository — staying on the fork's locked manifest keeps the environment frozen until the instructor is ready to move forward.

> Important: `swift package update` resolves dependencies past the pinned versions in the manifest. Avoid running it during a course unless the goal is explicitly to upgrade Quiver — the regular `swift run` and `swift build` commands respect the pin.

### Privacy and network behavior

The privacy story is covered in detail in <doc:Quiver-Notebook>. The short version for adoption decisions: nothing leaves the machine. The local server only accepts connections from the same machine, and there are no accounts, telemetry endpoints, or analytics calls. Bundled datasets ship with the repository and are read locally; custom CSVs loaded from disk stay local too.

One supported-configuration note: the Notebook is designed for one student per laptop, and runs Swift with the permissions of whoever launched it. Running it on a shared classroom server is not a supported configuration — a multi-user deployment would need sandboxing, resource limits, and per-user isolation that the current scope does not include.

### What if port 8080 is in use

The Notebook listens on port `8080` by default. If another process is already using that port — common on machines that also run a local development server — we set the `PORT` environment variable before launching:

```bash
PORT=8090 swift run
```

The active port is logged to the terminal at startup, and the browser tab needs to point at the same one (`http://localhost:8090` for the example above). Any unused TCP port works.

> Tip: Changing the port does not change the bind address. The server still listens only on `127.0.0.1` and refuses connections from elsewhere on the network — a custom port is a convenience choice, not a way to expose the Notebook to other machines.

### Carrying student work into apps

The most direct payoff for a course aimed at Apple-platform engineers: students prototype in the Notebook, then carry the same code into an Xcode project for the app-development unit. There is no separate notebook language and no hidden cell behavior to translate away — the only difference between a snippet here and the same code in an app is that the app writes its own `import Quiver` at the top of each file, where the Notebook supplies that line for us.

For saving a trained model from the Notebook and loading it inside an app, see <doc:Model-Persistence>. For the model APIs themselves, see <doc:Linear-Regression>, <doc:Naive-Bayes>, <doc:Nearest-Neighbors-Classification>, and <doc:KMeans-Clustering>.

### Related
- <doc:Quiver-Notebook>
- <doc:Notebook-Datasets>
- <doc:Model-Persistence>
- <doc:Panel>
