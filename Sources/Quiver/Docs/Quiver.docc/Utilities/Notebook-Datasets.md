# Notebook Datasets

A library of teaching datasets bundled into the Quiver Notebook.

## Overview

The Quiver Notebook ships with a small library of teaching datasets accessible by name from the editor. Iris, Titanic, California Housing, and a handful of others load with a single line of code, each paired with a target column for either classification or regression. A separate loader reads any CSV from disk in the same shape, so a class can move from bundled data to its own data without changing the rest of a snippet.

The dataset library is part of the Notebook itself rather than the Quiver package. This page covers the menu of bundled datasets, how to load and inspect them, and how to pull in a CSV from disk. For starting the Notebook and writing snippets, see <doc:Quiver-Notebook>.

### How to load a dataset

Loading is a single line: `Dataset.iris` returns the iris dataset, `Dataset.titanic` returns the Titanic passenger manifest, and so on. Because the underlying file could in principle be missing or unreadable, each accessor returns an optional value, so we begin with `guard let` to confirm the load succeeded:

```swift
guard let iris = Dataset.iris else {
    print("Couldn't load Dataset.iris.")
    exit(1)
}
```

If the load fails, the `else` branch prints a message and calls `exit(1)` to end the program with a failure status. The remaining snippets on this page use the same guard.

The `Dataset` type is the name we use to group the bundled datasets together. Typing `Dataset.` in the editor brings up an autocomplete menu of every available dataset. The values that come back are safe to pass into SwiftUI views or background work without extra ceremony.

### Inspecting a dataset

The standard pattern for getting oriented in a new dataset is three steps: load it, pull out the underlying table of values, and print a few rows:

```swift
guard let iris = Dataset.iris else {
    print("Couldn't load Dataset.iris.")
    exit(1)
}

let panel = iris.toPanel()
print(panel.head(n: 3))
print(iris.description)
print("shape:", panel.shape)
```

The `toPanel()` method returns the dataset as a Quiver <doc:Working-With-Panels>: a table of named columns where every column is a vector of `Double` values. The `head(n:)` method returns the first few rows as a formatted string, so we wrap it in `print` to see what loaded. The `description` property is a one-paragraph summary of where the dataset came from, what its columns mean, and any cleaning that was applied before bundling, and it is useful to read out loud at the start of a lecture. The `shape` property returns the row and column counts as a tuple, so a `print` confirms the load matches expectations before any modeling work begins.

### The bundled tabular datasets

Five tabular datasets ship with the Notebook, alongside two time-series sets covered below. Each tabular dataset comes with a target column suitable for either regression or classification. None of the bundled datasets contains missing values; any gaps in the original sources were resolved during cleaning, and each dataset's `description` says how.

The `Dataset.iris` accessor returns 150 rows, 5 columns. Classification. Label column: `species` (three classes, encoded as described in the categorical columns section below). Four flower measurements per row, originally collected by Edgar Anderson and published by R. A. Fisher in 1936. Useful for teaching a first classifier on well-separated classes.

The `Dataset.titanic` accessor returns 889 rows, 8 columns. Classification. Label column: `Survived` (0/1). Cleaned passenger manifest from the 1912 disaster. Useful for teaching mixed numeric and categorical features and class-imbalance trade-offs against a familiar binary outcome. The cleaning filled the 177 missing ages with the median age of 28 and dropped the two rows with no port of embarkation. As a result, 202 passengers have an age of exactly 28, a spike that shows up in any histogram of the `Age` column and is worth pointing out before a lesson on distributions.

The `Dataset.californiaHousing` accessor returns 20,640 rows, 10 columns. Regression. Target column: `median_house_value`. 1990 California census districts. Useful for teaching feature scaling, geographic features, and how far a linear model gets on real-world tabular data.

The `Dataset.bikeSharing` accessor returns 731 rows, 16 columns. Regression. Target column: `cnt` (total daily rides). Daily Capital Bikeshare ride counts for 2011 and 2012, paired with weather, calendar, and season features. Two leakage traps live in this dataset. First, the `casual` and `registered` columns add up to `cnt` exactly on every row, so a model that includes them predicts the target perfectly without learning anything; leave both out of the feature columns. Second, the rows form a time series, so a shuffled split lets the model train on days that come after the days it is tested on. The time-aware split section below shows how to hold out the later rows instead.

The `Dataset.studentPerformance` accessor returns 395 rows, 33 columns. Regression (or classification on a thresholded `G3`). Target column: `G3` (final grade, 0–20). Portuguese secondary-school students with family, study, and lifestyle features alongside three sequential grade columns (G1, G2, G3). Useful for teaching feature selection and the leakage trap of training on G1 and G2 to predict G3.

### The bundled time-series datasets

Two synthetic time-series datasets ship alongside the tabular set. Both describe a single 60-second tempo run for a moderately trained 70 kg runner, simulated against published physiological constraints. Real wearable recordings are messy and carry privacy concerns, so a synthetic run gives a class a clean, shareable starting point for multi-rate sensor work.

The `Dataset.simulatedRun` accessor returns 60 rows, 6 columns, sampled at 1 Hz. No label column. Across the minute, pace and power hold flat while `heartRate` climbs by roughly 6 BPM. This is heart rate catching up with a steady workload at the start of an effort: the body's response lags the work being done. It is not cardiac drift, the slow heart-rate rise that unfolds over efforts of tens of minutes at a constant pace. Even in one minute, the gap shows why a single signal can misread effort and why reading several signals together gives a fuller picture.

The `Dataset.simulatedRunAccel` accessor returns 3,000 rows, 2 columns, sampled at 50 Hz. No label column. The `magnitude` signal combines a step-frequency fundamental at 2.833 Hz (170 steps per minute, counting both feet), its second harmonic at 5.667 Hz, and Gaussian noise around a 1g baseline. The 50 Hz rate is fast enough to resolve both frequencies, which makes the dataset a clean starting point for power-spectral-density analysis, dominant-frequency detection, and band-energy features for activity classification.

The same load-and-inspect pattern from earlier in this article carries over directly:

```swift
guard let run = Dataset.simulatedRun else {
    print("Couldn't load Dataset.simulatedRun.")
    exit(1)
}

let panel = run.toPanel()
print(panel.head(n: 3))
print("shape:", panel.shape)
```

Both datasets describe the same physical event at different sample rates, so a class can move from heart-rate response at 1 Hz to spectral analysis on motion at 50 Hz without leaving the run.

### Working with categorical columns

A model needs numbers, not strings. When the source CSV has a column of class names (like the species names in iris), the loader converts them into numeric class indices the moment the dataset is read. Names sort alphabetically: `setosa` becomes `0`, `versicolor` becomes `1`, `virginica` becomes `2`. The original ordered names are kept in a separate dictionary so a predicted index can be turned back into a label for display:

```swift
guard let iris = Dataset.iris else {
    print("Couldn't load Dataset.iris.")
    exit(1)
}

let panel = iris.toPanel()
let species = panel["species"]

if let mapping = iris.categoricalMappings["species"] {
    let firstIndex = Int(species[0])  // 0
    let firstName = mapping[firstIndex]  // "setosa"
    print(firstName)
}
```

Numeric columns are left alone and do not appear in the categorical mapping.

### Handling missing values

The bundled datasets arrive complete, but a CSV loaded from disk often does not. When `Dataset.load(path:)` meets an empty cell, it stores `Double.nan` (the IEEE "not a number" value) rather than quietly filling it in with the column average. The reasoning is teaching: missing data should remain visible to students as part of the modeling process, not a detail the loader hides.

> Important: A `Double.nan` value spreads through sums, means, and distances: any of these that touches a `nan` also returns `nan`. Not every operation behaves this way, though. Comparison-based results such as a minimum, maximum, or median can step past a `nan` and return a normal-looking number, so a clean result is not proof that a column is clean. Check for missing values before fitting a model, and decide explicitly how to handle them. Quiver does not impute behind the scenes.

Because `nan` is the one value that never equals itself, comparing a column with itself produces a mask that is `true` wherever a value is present and `false` wherever it is missing. We can pass that mask to `filtered(where:)` to keep only the complete rows:

```swift
guard let survey = Dataset.load(path: "~/Desktop/survey.csv") else {
    print("Couldn't load survey.csv.")
    exit(1)
}

let panel = survey.toPanel()
let age = panel["age"]
let present = age.isEqual(to: age)

let complete = panel.filtered(where: present)
print("rows kept:", complete.shape.rows, "of", panel.shape.rows)
```

To fill the gaps with a chosen value instead of dropping rows, the same mask works with `choose(where:otherwise:)`, which keeps the original value where the mask is `true` and takes the fallback where it is `false`:

```swift
let fallback = [Double](repeating: 30.0, count: age.count)
let filledAge = age.choose(where: present, otherwise: fallback)
```

Whichever approach we pick, the choice is visible in the code, which is the point.

### A full classification pipeline

The bundled datasets work directly with Quiver's training and evaluation methods. A typical workflow on iris splits the data into a training set and a test set, trains a model on the training set, and reports how often the model is right on the held-out test set:

```swift
guard let iris = Dataset.iris else {
    print("Couldn't load Dataset.iris.")
    exit(1)
}

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

The label column lives in the same panel as the features, so the feature list must name its columns explicitly. Passing every column name, as in `toMatrix(columns: panel.columnNames)`, would hand the model the answer it is supposed to predict.

The `trainTestSplit(testRatio:seed:)` method shuffles the rows and then cuts them into two sets. The same seed always produces the same split, so every student in a class who runs the snippet with `seed: 42` gets the same rows and the same accuracy. The split is not stratified: it does not try to keep the class proportions equal between the two sets, which matters on an imbalanced dataset like Titanic, where a small test set can end up with noticeably more or fewer survivors than the full manifest.

The same pattern works for every classifier and regressor in `Quiver`: substitute the target column name, the feature columns, and the model type. See <doc:Naive-Bayes>, <doc:Linear-Regression>, and <doc:Nearest-Neighbors-Classification> for model-specific details.

### A time-aware split

Shuffling suits datasets where the rows are independent, but it leaks information on a time series. For bike sharing, a shuffled split puts days from late 2012 into the training set and tests the model on days from early 2011, so the model sees the future. A time-aware split trains on the earlier rows and tests on the later ones. The `yr` column (0 for 2011, 1 for 2012) gives a natural cut:

```swift
guard let bikes = Dataset.bikeSharing else {
    print("Couldn't load Dataset.bikeSharing.")
    exit(1)
}

let panel = bikes.toPanel()
let train = panel.filtered(where: panel["yr"].isLessThan(1.0))
let test = panel.filtered(where: panel["yr"].isGreaterThanOrEqual(1.0))
print("train:", train.shape.rows, "test:", test.shape.rows)  // train: 365 test: 366
```

Training on 2011 and testing on 2012 also exposes a real-world effect: ridership grew between the two years, so a model fit on the first year underpredicts the second.

### Word embeddings

A word embedding represents each word as a vector of numbers, arranged so that words with similar meanings sit close together in vector space. The bundled embeddings cover the 25,000 most-frequent English words from Stanford's GloVe corpus, each represented as a 50-dimensional vector. Each row carries the `word`, its frequency `rank`, the vector `magnitude`, and the fifty components `dim_01` through `dim_50`. Words are looked up by string, and the vocabulary is lowercase only: `"paris"` is present, `"Paris"` is not. Less common words fall outside the 25,000-word slice, so every lookup returns an optional or an empty result rather than assuming the word exists.

Embeddings ship with the Notebook because the alternative is a multi-gigabyte download from a research site, a parser to write, and a long wait before the first lesson can begin. The 25,000-word slice is small enough to load quickly on a student's laptop and large enough to show the properties that matter: related words cluster, unrelated words sit far apart, and some analogies resolve as vector arithmetic. A class can move from "what is a vector" to "search by meaning" in the same session.

Once the dataset is loaded, the entire Quiver similarity surface applies directly to the returned vectors. The same vectors also feed clustering and document-search workflows with no conversion step.

> Tip: For the math and patterns these embeddings plug into, see <doc:Similarity-Operations> for cosine similarity and pairwise comparison, <doc:Semantic-Search> for end-to-end document ranking, and <doc:Text-Tokenization> for turning raw text into the tokens that look up these vectors.

Look up a single word's vector with the subscript:

```swift
guard let glove = Dataset.glove50d else {
    print("Couldn't load Dataset.glove50d.")
    exit(1)
}

if let king = glove["king"] {
    print(king.count)  // 50
}
```

The `nearest(to:k:)` method returns the `k` words whose vectors are closest to a given word by cosine similarity, excluding the query word itself:

```swift
for hit in glove.nearest(to: "paris", k: 3) {
    print("\(hit.rank). \(hit.word)  \(hit.score)")
}
// 1. france     0.80
// 2. brussels   0.78
// 3. amsterdam  0.78
```

The `analogy(_:_:_:k:)` method evaluates the word-analogy pattern "a is to b as c is to what?" It computes the vector `a − b + c`, then returns the `k` words closest to that point by cosine similarity, excluding the three input words. For `analogy("king", "man", "woman")`, the target vector is king − man + woman:

```swift
for hit in glove.analogy("king", "man", "woman", k: 3) {
    print("\(hit.rank). \(hit.word)  \(hit.score)")
}
// 1. queen     0.86
// 2. daughter  0.77
// 3. prince    0.76
```

Both methods return ranked tuples: rank starts at 1, paired with a cosine similarity score. The results are deterministic, so the same call always returns the same words. Not every analogy resolves this cleanly. The call `analogy("paris", "france", "berlin")` returns `"vienna"`, `"soho"`, and `"budapest"` as its top three, with `"germany"` nowhere among them, even though `"germany"` is in the vocabulary. Trying a few analogies and seeing which ones hold is a good classroom exercise.

When a single closest word is all that is needed, `nearestWord(of:)` returns just that word (a thin convenience over `nearest(to:k:)` with `k` of 1) and returns `nil` for a word outside the vocabulary:

```swift
glove.nearestWord(of: "paris")  // "france"
```

The `magnitude(of:)` method returns the length of a word's vector, the same value stored in the `magnitude` column, or `nil` when the word is absent:

```swift
glove.magnitude(of: "king")  // 5.37
```

### Loading a CSV from disk

When a class needs to use its own data, `Dataset.load(path:)` reads any CSV file from disk using the same parsing rules as the bundled datasets. The loader expands `~` to the user's home directory, so the path below points at a file saved to the Desktop:

```swift
// The path below is illustrative; replace it with the path to a CSV
// saved on your own machine.
guard let myData = Dataset.load(path: "~/Desktop/your-file.csv") else {
    print("Couldn't load your-file.csv.")
    exit(1)
}

let panel = myData.toPanel()
print(panel.head())
print("shape:", panel.shape)
```

The resulting dataset takes its name from the file name without the extension. Each column in the CSV is read according to its type. Numeric columns copy straight across as numbers. Boolean columns become `1.0` for true and `0.0` for false. String columns become numeric class indices the same way iris's `species` column does, with the original strings preserved in the categorical mapping. Date columns convert to Unix timestamps (the number of seconds since January 1, 1970), so a calendar date becomes a single number a model can work with. A timestamp loses the calendar structure (day of week, month, season) that often carries the signal, so when those matter, add them as separate columns in the CSV before loading, as the bike-sharing dataset does with its `weekday`, `mnth`, and `season` columns. Missing cells become `Double.nan`, as described in the missing values section above. Any column whose values do not match one of these types is skipped with a warning rather than failing the entire load.

The `Dataset.load(path:)` method returns `nil` in two cases: when the CSV cannot be parsed at all, and when it parses but none of its columns has a usable type. In both cases the underlying cause is written to standard error so it is visible in the Notebook's console pane.

> Tip: Skipped columns produce a warning on standard error but the load still succeeds with the remaining columns. Check `columnNames` after loading a custom CSV to confirm every expected column made it in. A silently dropped column is a common cause of "my model is missing a feature."

### Catalog

The `Dataset.catalog()` method returns a newline-separated listing of every bundled accessor, which is useful at the start of an exploration session or in a class handout that prints the full menu:

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

> Experiment: **The Quiver Notebook** is a good place to see how far the same three-line pattern carries. Run `load → toPanel → head(n: 3)` on `Dataset.iris`, then swap to `Dataset.titanic`, then `Dataset.californiaHousing`, then `Dataset.studentPerformance`. Each call returns a different shape (150 rows of flower measurements, 889 passengers, 20,640 census districts, 395 students) through the same API. Watching one snippet describe four datasets shows what `Panel` and `Dataset` are for. See <doc:Quiver-Notebook>.

### Related
- <doc:Quiver-Notebook>
- <doc:Working-With-Panels>
- <doc:Panel-Workflows>
- <doc:Train-Test-Split>
- <doc:Similarity-Operations>
- <doc:Semantic-Search>
