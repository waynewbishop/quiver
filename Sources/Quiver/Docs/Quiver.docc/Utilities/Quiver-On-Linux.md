# Quiver on Linux

Building and testing Quiver in a Linux container, and checking what a package supports.

## Overview

Quiver is a pure Swift library with no external dependencies, so it builds on Linux with nothing beyond a Swift toolchain. Its Linux CI workflow builds and tests it on Swift 5.9, the minimum toolchain, and on a current release. Docker lets us reproduce that environment from a Mac or any other development machine, which is the quickest way to confirm a change works before it reaches CI.

The two examples below answer different questions. A bind mount gives fast iteration against the working tree. A copy inside the container checks that the source behaves on a case-sensitive filesystem.

### Test the working tree

With Docker running, we mount the current checkout into a Swift container and run the same build and test steps as CI:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace swift:6.4 \
  swift test --scratch-path /tmp/quiver-linux --skip QuiverStressTests
```

The **scratch path** is the directory where Swift Package Manager writes build products. By default that is `.build` inside the package, which the mount shares with the host. A macOS build and a Linux build cannot reuse each other's products, so sharing one directory makes each side rebuild or fail. Pointing `--scratch-path` at a location inside the container keeps the Linux products out of the checkout entirely. Swapping `swift:6.4` for `swift:5.9` runs the minimum supported toolchain.

> Note: The stress tests run on demand only, matching local development and the CI workflow, so both examples skip `QuiverStressTests`.

### Check filename case

macOS volumes are usually case-insensitive, so `Sample.csv` and `sample.csv` name the same file there. On Linux they are two different files. Code that opens a file with the wrong capitalization passes on a Mac and fails in the container with a file-not-found error.

A bind mount does not catch this, because Docker Desktop on a Mac exposes the host filesystem's behavior to the container. To test against Linux's own rules, we copy the committed files into the container instead:

```bash
git archive HEAD | docker run --rm -i -w /workspace swift:6.4 bash -ec '
  tar -xf -
  swift test --skip QuiverStressTests
'
```

`git archive` writes the files exactly as Git records them, and `tar` unpacks them onto the container's case-sensitive filesystem. When a test cannot find a file, we compare the path in the code with the spelling Git records using `git ls-files`.

> Important: `git archive HEAD` copies only committed files. Uncommitted edits are absent from the container, so we commit or stash first, or use the bind-mount example for work in progress.

### Verify Linux support

Quiver's [Linux Tests workflow](https://github.com/waynewbishop/quiver/actions/workflows/linux.yml) lists the Swift versions it covers and the build and test result for each run. A green run on a release branch is the evidence that the version we plan to adopt works on Linux.

For any other Swift package, the same two sources apply. First, its CI results show whether anyone builds it on Linux at all. Second, a build and test run in our own container confirms it against the toolchain we deploy with. The `platforms` list in `Package.swift` is not that evidence. It declares minimum deployment versions for Apple platforms only, and a package that lists them can still build on Linux. See Apple's [Package Manager documentation](https://docs.swift.org/package-manager/PackageDescription/PackageDescription.html) for the manifest details.

### Running in Kubernetes

Because Quiver is pure Swift with no dependencies, we expect it to run in a Kubernetes container out of the box, the same way it runs in the `swift` Docker image.

### Where to go from here

See <doc:Installation> for adding Quiver to a package, <doc:How-It-Works> for why it carries no dependencies, and <doc:Quiver-Cookbook> for worked examples that run in any of these environments.
