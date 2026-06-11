# Retrieving Context for Generation

Assembling retrieved context blocks from chunked documents to ground a language model's answer.

## Overview

A language model answers from what it was trained on. To answer from a specific document — a user's notes, a product manual, a local knowledge base — the relevant passages have to be found first and handed to the model as context. That find-then-answer pattern is **retrieval-augmented generation**: retrieval pulls the passages that bear on a question, and the model generates an answer grounded in them.

Quiver owns the retrieval half. It indexes document fragments, ranks them against a query, and hands back a clean context block; the generation step stays with the model, and the math stays type-safe and dependency-free. The handoff between the two halves is a plain string — the most portable seam there is. This article builds that retrieval half on-device, from the same primitives <doc:Semantic-Search> and <doc:Embedding-Sources> already cover.

> Note: This article builds on <doc:Semantic-Search>, which teaches the embedding-and-ranking pipeline, and <doc:Embedding-Sources>, which defines the swappable source of vectors. The examples here are self-contained, but the vocabulary of vectors and similarity carries over. The embedding values shown are illustrative, chosen so the ranking is easy to follow.

## Splitting documents into chunks

A whole document is the wrong unit to retrieve. Averaging a long manual into one vector blurs every distinct passage into a single muddy point, and the answer to a specific question lives in one paragraph, not in the book's overall gist. So the first step is to split each document into **chunks** — fragments small enough to retrieve precisely, large enough to carry a coherent thought.

Quiver ships no chunker, because chunking is a domain decision: where to cut, how large to make each piece, and how much neighboring chunks should overlap all depend on the documents and the questions. The split is plain Swift over the text. A paragraph-aware splitter is the honest baseline — it cuts on structure the writing already has, rather than slicing mid-sentence at a fixed character count:

```swift
// A chunk carries provenance, not just text, so a retrieved fragment
// can be attributed back to its source. Sendable lets it cross the task
// boundaries a server handler or a background app task introduces.
struct Chunk: Codable, Sendable {
    let sourceID: String
    let index: Int
    let text: String
}

// Split a document on paragraph boundaries into provenance-tagged chunks.
func chunked(_ document: String, sourceID: String) -> [Chunk] {
    let paragraphs = document.components(separatedBy: "\n\n")
    var chunks: [Chunk] = []
    var index = 0
    for paragraph in paragraphs {
        let trimmed = paragraph.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { continue }
        chunks.append(Chunk(sourceID: sourceID, index: index, text: trimmed))
        index += 1
    }
    return chunks
}
```

The `Chunk` carries more than its text. A retrieved fragment has to be attributable — which document it came from, and where in it — so the assembled context can cite its sources and a reader can trust where the answer is grounded. That is the one field this needs beyond the `(text, vector)` pairing in <doc:Embedding-Sources>: a provenance tag that survives all the way to the final block.

Paragraph splitting is the baseline, not the ceiling. Two refinements matter in practice and are worth naming. Small chunks retrieve precisely but fragment context; large chunks keep context coherent but dilute the similarity score and waste the model's limited input budget. And a fixed split can strand an answer that straddles a boundary, so production splitters carry a small overlap from one chunk into the next. Both are tuning decisions for the developer's own documents; the math downstream does not change when the split does.

## Embedding the chunks once

Retrieval is a write-once, read-many pattern. The chunks are embedded a single time when a document enters the system, the vectors are stored, and only the query is embedded at search time. Embedding every chunk on every query would pay the expensive step repeatedly for a result that never changes.

We embed through a `some Embedder` source — the contract <doc:Embedding-Sources> defines — so the vectors can come from a hand-built table while learning the pipeline and from an on-device sentence model in production, with no change to the code around them. Each chunk's vector is stored alongside the chunk itself, so provenance rides along into the index:

```swift
import Quiver

// `embedder` is any `some Embedder`; the chunks come from `chunked(_:sourceID:)`.
let chunks = chunked(document, sourceID: "bread-guide")

// Embed each chunk once, at ingest. Keep the chunks and their vectors aligned by index.
var storedChunks: [Chunk] = []
var storedVectors: [[Double]] = []
for chunk in chunks {
    if let vector = embedder.embed(chunk.text) {
        storedChunks.append(chunk)
        storedVectors.append(vector)
    }
}
```

The chunks are plain `Codable` values and the vectors are plain `[Double]`, so the index persists to disk and loads at the next launch — a document is embedded once in its lifetime and queried for free thereafter. See <doc:Model-Persistence> for the encode-and-decode shape this index shares with every persisted value.

## Retrieving the relevant chunks

With the index built, retrieval is the ranking pipeline <doc:Semantic-Search> teaches, run against the stored vectors. The query is embedded through the same source, `cosineSimilarities(to:)` scores every chunk in one call, and `topIndices(k:labels:)` reads off the closest few — carrying each chunk through as the label, so provenance survives the ranking without a second lookup:

```swift
import Quiver

// Only the query is embedded at search time — the chunks were embedded at ingest.
guard let queryVector = embedder.embed("how long should the dough rise") else {
    return
}

let scores = storedVectors.cosineSimilarities(to: queryVector)
let hits = scores.topIndices(k: 2, labels: storedChunks)
// hits[0]: (rank: 1, label: Chunk(bread-guide, 0, …), score: 0.9939)
// hits[1]: (rank: 2, label: Chunk(bread-guide, 1, …), score: 0.8046)
```

Passing the chunks as `labels` is what keeps retrieval honest: each hit arrives already holding the `Chunk` that produced its score — text and provenance — with no parallel array to index back into. The raw cosine value is the right thing to keep here, not a percentage: it is a similarity, and a retrieval system uses it as a threshold for whether a chunk is relevant enough to include, not as a confidence that the answer is correct.

## Assembling the context block

The retrieved chunks become a single **context block** — the formatted text handed to the model. Assembly is plain string work: walk the hits and join each chunk's text with its provenance so the model can attribute what it reads. A length budget keeps the block inside the model's input limit:

```swift
// Build the context block from the ranked hits — each carries its own chunk.
var block = ""
for hit in hits {
    let chunk = hit.label
    block += "[\(chunk.sourceID)#\(chunk.index)] \(chunk.text)\n\n"
}
// [bread-guide#0] Let the dough rise slowly. A slow proof develops flavor…
// [bread-guide#1] Knead the dough until smooth. Good kneading builds…
```

The block is bounded by how many chunks we include and how large each one is — the two ends of the chunking tradeoff named earlier. A model reads a fixed amount of text at once, its **context window**, so retrieval earns its place by selecting the few fragments most worth that budget rather than handing over the whole document.

## The retrieval boundary

The context block is the last thing Quiver produces. It is a plain string, ready to hand to an on-device language model — the generation half that retrieval pairs with. Quiver does not generate, does not call a model, and does not need to: the boundary between retrieval and generation is the string itself, which is what makes the retrieval layer droppable into any stack. The model can change without a single line of the retrieval code changing with it.

> Important: Quiver produces the context block; the model consumes it; the handoff is a string. Retrieval ranks and assembles — it never generates. Building the block is the work shown here; turning it into a prompt and generating an answer belongs to whatever model the app runs.

That division is the same separation of concerns the rest of the platform follows: a source of vectors distills meaning, Quiver indexes and ranks it, and the surface beyond the boundary — a view, a model, a response — consumes a clean value. For where an on-device feature reads that value, see <doc:iOS-Apps>.

> Experiment: **The Quiver Notebook** is the right place to watch retrieval narrow a document to its relevant fragments. Load `Dataset.glove50d`, chunk a short multi-paragraph passage, and retrieve the top fragments for a question — then change the question and re-run. The same chunks re-rank, a different block assembles, and the fragment that answers the new question rises to the top. That shift is retrieval doing its one job. See <doc:Quiver-Notebook>.
