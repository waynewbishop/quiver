# Notebook Datasets

Working with the datasets bundled into the Quiver Notebook.

## Overview

The Quiver Notebook ships with a small library of teaching datasets accessible by name from the editor, with no download, no parsing, and no setup. Iris, Titanic, California Housing, and a handful of others are ready to load with a single line of code, paired with sensible target columns for either classification or regression. A separate loader reads any CSV from disk in the same shape, so a class can move from bundled data to its own data without changing the rest of a snippet.

The dataset library is part of the Notebook itself rather than the Quiver package. This page covers the menu of bundled datasets, how to load and inspect them, and how to pull in a CSV from disk. For starting the Notebook and writing snippets, see <doc:Quiver-Notebook>.

### How to load a dataset

Loading is a single line: `Dataset.iris` returns the iris dataset, `Dataset.titanic` returns the Titanic passenger manifest, and so on. Because the underlying file could in principle be missing or unreadable, each accessor returns an optional value, so we begin with `guard let` to confirm the load succeeded:

```swift
guard let iris = Dataset.iris else {
    exit(0)
}
```

The `Dataset` type is the name we use to group the bundled datasets together. Typing `Dataset.` in the editor brings up an autocomplete menu of every available dataset. The values that come back are safe to pass into SwiftUI views or background work without extra ceremony.

> Tip: For a printable list of every bundled accessor without leaving the editor, call `Dataset.catalog()`, which returns a newline-separated listing suitable for a class handout or an exploration session.

### Inspecting a dataset

The standard pattern for getting oriented in a new dataset is three steps: load it, pull out the underlying table of values, and print a few rows:

```swift
guard let iris = Dataset.iris else {
    exit(0)
}

let panel = iris.toPanel()
print(panel.head(n: 3))
print(iris.description)
print("shape:", panel.shape)
```

The `toPanel()` method returns the dataset as a Quiver <doc:Panel>: a table of named columns where every column is a vector of `Double` values. The `head(n:)` method prints the first few rows so we can eyeball what loaded. The `description` property is a one-paragraph summary of where the dataset came from, what its columns mean, and any cleaning that was applied before bundling, and it is useful to read out loud at the start of a lecture. The `shape` property returns the row and column counts as a tuple, so a `print` confirms the load matches expectations before any modeling work begins.

### The bundled tabular datasets

Five tabular datasets ship with the Notebook alongside a paired time-series set covered below. Each tabular dataset is paired with a target column suitable for either regression or classification.

The `Dataset.iris` accessor returns 150 rows, 5 columns. Classification. Label column: `species` (encoded alphabetically as setosa→0, versicolor→1, virginica→2). The classic introductory classification dataset, originally collected by Edgar Anderson and published by R. A. Fisher in 1936.

The `Dataset.titanic` accessor returns 889 rows, 8 columns. Classification. Label column: `Survived` (0/1). Cleaned passenger manifest from the 1912 disaster. Useful for teaching mixed numeric and categorical features, missing-value handling, and class-imbalance trade-offs against a familiar binary outcome.

The `Dataset.californiaHousing` accessor returns 20,640 rows, 10 columns. Regression. Target column: `median_house_value`. 1990 California census districts. The standard introductory regression dataset for teaching feature scaling, geographic features, and the gap between linear and tree-based models on real-world tabular data.

The `Dataset.bikeSharing` accessor returns 731 rows, 16 columns. Regression. Target column: `cnt` (total daily rides). Daily Capital Bikeshare ride counts paired with weather, calendar, and season features. A clean introduction to time-aware regression and to the seasonality patterns that make naive splits leak information.

The `Dataset.studentPerformance` accessor returns 395 rows, 33 columns. Regression (or classification on a thresholded `G3`). Target column: `G3` (final grade, 0–20). Portuguese secondary-school students with family, study, and lifestyle features alongside three sequential grade columns (G1, G2, G3). Useful for teaching feature selection and the leakage trap of training on G1 and G2 to predict G3.

### The bundled time-series datasets

Two synthetic time-series datasets ship alongside the tabular set. Both come from a single 60-second tempo run for a moderately trained 70 kg runner, simulated against published physiological constraints. Real wearable recordings are messy and privacy-laden, and the HR-pace-decoupling-and-spectral-analysis pair is the cleanest small-data demonstration of multi-rate sensor work we can offer for teaching.

The `Dataset.simulatedRun` accessor returns 60 rows, 6 columns, sampled at 1 Hz. No label column. Across the minute, pace and power hold flat while `heartRate` drifts upward by roughly 6 BPM: the HR-pace-decoupling pattern that underlies True Effort Score work. The gap between the visible work the runner is doing and the body's rising response is precisely why a single-signal effort score misleads and a multi-signal one holds up. The dataset is the smallest honest demonstration we know of how one signal can mislead and several signals together reveal real effort.

The `Dataset.simulatedRunAccel` accessor returns 3,000 rows, 2 columns, sampled at 50 Hz. No label column. The `magnitude` signal combines a footstrike fundamental at 2.833 Hz (170 steps per minute), a two-legged-gait harmonic at 5.667 Hz, and Gaussian noise around a 1g baseline. The 50 Hz rate is fast enough to resolve gait frequencies, which makes the dataset a clean starting point for power-spectral-density analysis, dominant-frequency detection, and band-energy features for activity classification.

The same load-and-inspect pattern from earlier in this article carries over directly: `head(n:)` reveals the column names and a few rows, and `shape` confirms the row and column counts:

```swift
guard let run = Dataset.simulatedRun else {
    exit(0)
}

let panel = run.toPanel()
print(panel.head(n: 3))
print("shape:", panel.shape)
```

Both datasets describe the same physical event at different sample rates, so a class can move from HR lag at 1 Hz to spectral analysis on motion at 50 Hz without leaving the run.

### Working with categorical columns

A model needs numbers, not strings. When the source CSV has a column of class names (like the species names in iris), the loader converts them into numeric class indices the moment the dataset is read. Names sort alphabetically: `setosa` becomes `0`, `versicolor` becomes `1`, `virginica` becomes `2`. The original ordered names are kept in a separate dictionary so a predicted index can be turned back into a label for display:

```swift
guard let iris = Dataset.iris else { exit(0) }

let panel = iris.toPanel()
let species = panel["species"]

if let mapping = iris.categoricalMappings["species"] {
    let firstIndex = Int(species[0])               // 0
    let firstName  = mapping[firstIndex]           // "setosa"
    print(firstName)
}
```

Numeric columns are left alone. They do not appear in the categorical mapping. Missing values in the source CSV are preserved as `Double.nan` (the IEEE "not a number" value) rather than quietly filled in with the column average. The reasoning is teaching: missing data should remain visible to students as part of the modeling process, not a detail the loader hides behind their backs.

> Important: `Double.nan` propagates through arithmetic: any sum, mean, or distance involving a `nan` value also returns `nan`. Before fitting a model on a dataset with missing values, decide explicitly how to handle them: drop the affected rows with `outlierMask` and `masked(by:)`, or fill them with a chosen value. Quiver does not impute behind the scenes.

### A full classification pipeline

The bundled datasets work directly with Quiver's training and evaluation methods. A typical workflow on iris splits the data into a training set and a test set, trains a model on the training set, and reports how often the model is right on the held-out test set:

```swift
guard let iris = Dataset.iris else { exit(0) }

let panel = iris.toPanel()
let featureColumns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

let (train, test) = panel.trainTestSplit(testRatio: 0.2, seed: 42)
let model = GaussianNaiveBayes.fit(
    features: train.toMatrix(columns: featureColumns),
    labels: train.labels("species")
)

let predictions = model.predict(test.toMatrix(columns: featureColumns))
let accuracy = predictions.confusionMatrix(actual: test.labels("species")).accuracy
print("accuracy:", accuracy)
```

The same pattern works for every classifier and regressor in `Quiver`: substitute the target column name, the feature columns, and the model type. See <doc:Naive-Bayes>, <doc:Linear-Regression>, and <doc:Nearest-Neighbors-Classification> for model-specific details.

### Word embeddings

A word embedding represents each word as a vector of numbers, arranged so that words with similar meanings sit close together in vector space. The bundled embeddings cover the 25,000 most-frequent English words from Stanford's GloVe corpus, each represented as a 50-dimensional vector. Each row carries the `word`, its frequency `rank`, the vector `magnitude`, and the fifty components `dim_01` through `dim_50`. Words are looked up by string.

Embeddings ship with the Notebook because the alternative is a multi-gigabyte download from a research site, a parser to write, and a ten-minute wait before the first lesson can begin. The 25,000-word slice is small enough to load quickly on a student's laptop and large enough to demonstrate every property that matters: synonyms cluster, analogies hold, and unrelated words sit far apart. A class can move from "what is a vector" to "search by meaning" in the same session.

Once the dataset is loaded, the entire Quiver similarity surface applies directly to the returned vectors. The same vectors also feed clustering and document-search workflows with no conversion step.

> Tip: For the math and patterns these embeddings plug into, see <doc:Similarity-Operations> for cosine similarity and pairwise comparison, <doc:Semantic-Search> for end-to-end document ranking, and <doc:Text-Tokenization> for turning raw text into the tokens that look up these vectors.

Look up a single word's vector with the subscript:

```swift
guard let glove = Dataset.glove50d else { exit(0) }

if let king = glove["king"] {
    print(king.count)  // 50
}
```

The `nearest(to:k:)` method returns the `k` words whose vectors are closest to a given word by cosine similarity, excluding the query word itself:

```swift
for hit in glove.nearest(to: "paris", k: 3) {
    print("\(hit.rank). \(hit.word)  \(hit.score)")
}
// 1. france    0.80
// 2. brussels  0.78
// 3. french    0.77
```

The `analogy(_:_:_:k:)` method evaluates the classic word-analogy pattern. The call `analogy("king", "man", "woman", k: 1)` typically returns `"queen"`. Both methods return ranked tuples: rank starts at 1, paired with a cosine similarity score.

When a single closest word is all that is needed, `nearestWord(of:)` returns just that word (a thin convenience over `nearest(to:k:)` with `k` of 1) and returns `nil` for a word outside the vocabulary:

```swift
glove.nearestWord(of: "paris")  // "france"
```

The `magnitude(of:)` method returns the length of a word's vector, the same value stored in the `magnitude` column, or `nil` when the word is absent:

```swift
glove.magnitude(of: "king")  // 5.37
```

For the underlying vector math, see <doc:Similarity-Operations> and <doc:Semantic-Search>.

### Loading a CSV from disk

When a class needs to use its own data, `Dataset.load(path:)` reads any CSV file from disk using the same parsing rules as the bundled datasets. The loader expands `~` to the user's home directory, so a path like the one below points at a file the student saved to their Desktop:

```swift
// The path below is illustrative — replace it with the path to a CSV
// you have saved on your own machine.
guard let myData = Dataset.load(path: "~/Desktop/your-file.csv") else {
    exit(0)
}

print(myData.toPanel().head())
print("shape:", myData.shape)
```

The resulting dataset takes its name from the file name without the extension. Each column in the CSV is read according to its type. Numeric columns copy straight across as numbers. String columns become numeric class indices the same way iris's `species` column does, with the original strings preserved in the categorical mapping. Date columns convert to Unix timestamps (the number of seconds since January 1, 1970), so a calendar date becomes a single number a model can work with. Missing cells become `Double.nan`. Any column whose values do not match one of these types is skipped with a warning rather than failing the entire load.

The `Dataset.load(path:)` method returns `nil` only if the CSV cannot be parsed at all. The underlying error is written to standard error so the cause is visible in the Notebook's console pane.

> Tip: Skipped columns produce a warning on standard error but the load still succeeds with the remaining columns. Check `columnNames` after loading a custom CSV to confirm every expected column made it in. A silently dropped column is a common cause of "my model is missing a feature."

### Catalog

The `Dataset.catalog()` method returns a newline-separated listing of every bundled accessor. Useful at the start of an exploration session, or in a class handout that prints the full menu:

```swift
print(Dataset.catalog())
// Dataset.iris
// Dataset.titanic
// Dataset.californiaHousing
// Dataset.bikeSharing
// Dataset.studentPerformance
// Dataset.simulatedRun
// Dataset.simulatedRunAccel
// Dataset.glove50d
```

> Experiment: **The Quiver Notebook** is the right place to feel how far the same three-line pattern carries. Run `load → toPanel → head(n: 3)` on `Dataset.iris`, then swap to `Dataset.titanic`, then `Dataset.californiaHousing`, then `Dataset.studentPerformance`. Each call returns a wildly different shape — 150 rows of flower measurements, 889 passengers, 20,640 census districts, 395 students — through the same API. Watching the same snippet describe five datasets is the fastest way to see what `Panel` and `Dataset` are for. See <doc:Quiver-Notebook>.

### Related
- <doc:Quiver-Notebook>
- <doc:Panel>
- <doc:Panel-Workflows>
- <doc:Train-Test-Split>
- <doc:Similarity-Operations>
- <doc:Semantic-Search>
