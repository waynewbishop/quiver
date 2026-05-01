# Text Tokenization

Splitting strings into tokens and looking up word vectors before any downstream numeric work.

## Overview

Most numerical workflows assume the data has already been turned into numbers. Text is the exception. Before a classifier, a vector database, or a similarity comparison can act on a sentence, the sentence has to be broken into words and each word has to be mapped to a vector. Quiver provides two paired methods for that translation: `tokenize` on `String` for the splitting step, and `embed(using:)` on `[String]` for the lookup step.

The two methods compose to produce a `[[Double]]` from raw text — one inner array per recognized word. From there, the rest of Quiver's vector operations apply. The <doc:Semantic-Search> article walks the full search pipeline from tokenization through ranked results; this article is the closer focus on tokenization and embedding as a standalone preprocessing pair, useful any time a developer needs to feed text into a numeric workflow.

### Tokenizing strings

The `tokenize(strippingPunctuation:)` method on `String` lowercases the input, splits on whitespace and newlines, removes empty tokens, and — by default — strips leading and trailing punctuation from each token. The defaults are tuned for embedding lookups, where punctuated forms like `"shoes!"` and `"shoes,"` would otherwise miss their dictionary key:

```swift
import Quiver

let query = "Comfortable Running Shoes!"
let tokens = query.tokenize()
// ["comfortable", "running", "shoes"]

// Punctuation preserved when it carries meaning
let raw = "Hello, world!".tokenize(strippingPunctuation: false)
// ["hello,", "world!"]
```

Punctuation stripping operates on the boundaries of each token, so interior characters survive. Contractions like `"don't"` keep their apostrophe because the apostrophe is not at either end after the initial trim.

> Note: `tokenize` does not stem, lemmatize, or remove stop words. It is a literal split-and-clean step, leaving any vocabulary or normalization decisions to the caller.

### Looking up embeddings

Once the text is a `[String]`, the `embed(using:)` method maps each token through a dictionary of word vectors and returns the inner arrays for the words it finds. Words missing from the dictionary are silently filtered out, so the method is forgiving when the input contains rare tokens, function words, or typos:

```swift
let words = ["running", "shoes", "unknown"]
let embeddings: [String: [Double]] = [
    "running": [0.8, 0.7, 0.9],
    "shoes":   [0.1, 0.9, 0.2]
]

let vectors = words.embed(using: embeddings)
// [[0.8, 0.7, 0.9], [0.1, 0.9, 0.2]]
// "unknown" is filtered out
```

The dictionary is plain `[String: [Double]]` — Quiver does not assume where the vectors come from. They might originate in a pre-trained language model exported to JSON, a domain-specific table built from in-house data, or a remote service that returns vectors at boot. The contract is the dictionary type, not the source.

### Building a document vector

Tokenizing a sentence produces multiple word vectors, but most downstream tasks expect one vector per document. The standard reduction is `meanVector`, which averages the inner arrays element-by-element to produce a single `[Double]` representing the document's overall meaning. The full pipeline reads naturally as a chain:

```swift
let document = "Comfortable, lightweight running shoes."

let documentVector = document
    .tokenize()
    .embed(using: embeddings)
    .meanVector()
// Optional<[Double]> — nil if no recognized tokens
```

From this point, the result is an ordinary `[Double]` and the rest of Quiver applies — `cosineSimilarities(to:)` for ranking against a corpus, `distance(to:)` for nearest-neighbor work, or any other vector operation. The full search example, including building a corpus and ranking results, is in <doc:Semantic-Search>. The aggregation reference for `meanVector` and related summaries is in <doc:Statistical-Operations>.

