# Retrieving Context for Generation

Assembling retrieved context blocks from chunked documents to ground a language model's answer.

## Overview

Retrieval is a ranking problem. Split a document into passages, turn each one and the question into a vector, and the passages that answer the question are the ones whose vectors point most nearly the same way — measured by the cosine similarity of <doc:Semantic-Search>. Collecting the highest-scoring few into one block is the whole operation, and what reads that block is a separate choice: a language model answering from it is the **retrieval-augmented generation** (RAG) pattern, but a search result list, a report, or a citation set are the same ranking with a different reader. This article assembles that pipeline — split, embed, rank, collect — and keeps each passage tied to where it came from along the way.

> Note: This article builds on <doc:Semantic-Search>, which teaches the embedding-and-ranking pipeline, and <doc:Embedding-Sources>, which defines the swappable source of vectors. The examples here are self-contained, but the vocabulary of vectors and similarity carries over. The embedding values shown are illustrative, chosen so the ranking is easy to follow.

## Splitting documents into chunks

A whole document is too large to retrieve against directly, so the first step is to split each one into **chunks** — fragments small enough to retrieve precise information and large enough to carry a coherent thought.

Quiver ships no chunker, because chunking is a design decision. Where to cut, how large to make each piece, and how much neighboring chunks should overlap all depend on the documents being indexed. A paragraph-aware splitter is a good starting point: it cuts on the document's own structure rather than splitting sentences at arbitrary points.

```swift
// A chunk carries provenance, not just text, so a retrieved fragment
// can be attributed back to its source. Sendable lets it cross the task
// boundaries a server handler or a background app task introduces.
struct Chunk: Codable, Sendable {
    let sourceID: String
    let index: Int
    let text: String

    // The form handed to the model later — text tagged with its provenance,
    // e.g. "[bread-guide#0] Let the dough…". Keeping the format on the type
    // means the citation is one fact in one place, not re-spelled at assembly.
    var citedForm: String { "[\(sourceID)#\(index)] \(text)" }
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

// A chunk paired with its embedding. Binding the two in one value is what keeps
// labels honest: chunk and vector are structurally inseparable, so a later edit
// cannot silently desync two arrays kept aligned by hand. The chunk is also the
// readable form the vector cannot be read back into, so storing it here means a
// ranked hit already holds its text — nothing to recover from the index. Both
// fields are already Codable, so the whole index persists.
struct StoredChunk: Codable, Sendable {
    let chunk: Chunk
    let vector: [Double]
}

// `embedder` is any `some Embedder`; the chunks come from `chunked(_:sourceID:)`.
let chunks = chunked(document, sourceID: "bread-guide")

// Embed each chunk once, at ingest. Chunk and vector go into `stored` together —
// never as two arrays the code has to keep aligned itself.
var stored: [StoredChunk] = []
for chunk in chunks {
    if let vector = embedder.embed(chunk.text) {
        stored.append(StoredChunk(chunk: chunk, vector: vector))
    }
}
```

Each `StoredChunk` is a plain `Codable` value pairing the chunk with its `[Double]` vector — the same `(text, vector)` honesty <doc:Embedding-Sources> establishes, now carrying provenance too. An index that stored only the searchable vector would have to reconstruct the readable text on the way out; pairing the text with its vector means there is nothing to reconstruct — what we rank is what we read. So the index persists to disk and loads at the next launch: a document is embedded once in its lifetime and queried for free thereafter. See <doc:Model-Persistence> for the encode-and-decode shape this index shares with every persisted value.

## Retrieving the relevant chunks

With the index built, retrieval is the ranking pipeline <doc:Semantic-Search> teaches, run against the stored vectors. The query is embedded through the same source, then `cosineSimilarities(to:)` scores every chunk in one call. `topIndices(k:labels:)` reads off the closest few, carrying each chunk through as the label, so provenance survives the ranking without a second lookup:

```swift
import Quiver

// Only the query is embedded at search time — the chunks were embedded at ingest.
guard let queryVector = embedder.embed("how long should the dough rise") else {
    return
}

// `stored` is the single source of truth. Derive the parallel views the ranking
// needs — vectors to score, chunks to carry through as labels — both in `stored` order.
var vectors: [[Double]] = []
var chunks: [Chunk] = []
for item in stored {
    vectors.append(item.vector)
    chunks.append(item.chunk)
}

let scores = vectors.cosineSimilarities(to: queryVector)
let hits = scores.topIndices(k: 2, labels: chunks)
// hits[0]: (rank: 1, label: Chunk(bread-guide, 0, …), score: 0.9939)
// hits[1]: (rank: 2, label: Chunk(bread-guide, 1, …), score: 0.8046)
```

Passing the chunks as `labels` is what keeps retrieval honest: each hit arrives already holding the `Chunk` that produced its score — text and provenance — so there is no array to index back into after ranking. The vectors and chunks are derived from `stored` for this one call and discarded; `stored` stays the single source of truth, the pairing intact. The raw cosine value is the right thing to keep here, not a percentage: it is a similarity, and a retrieval system uses it as a threshold for whether a chunk is relevant enough to include, not as a confidence that the answer is correct.

## Assembling the context block

The retrieved chunks become a single **context block** — the formatted text handed to the model. Assembly is plain string work: walk the hits and join each chunk's `citedForm` — its text tagged with provenance — so the model can attribute what it reads. Because the citation format lives on `Chunk`, the loop never re-spells it. The block is bounded by `k` and by chunk size — the two ends of the chunking tradeoff — so it stays inside the model's input limit without a separate budget:

```swift
// Build the context block from the ranked hits — each carries its own chunk.
// The separator goes between entries, not after each, so the block carries no
// trailing blank line.
var block = ""
for hit in hits {
    let separator = block.isEmpty ? "" : "\n\n"
    block += separator + hit.label.citedForm
}
// [bread-guide#0] Let the dough rise slowly. A slow proof develops flavor…
//
// [bread-guide#1] Knead the dough until smooth. Good kneading builds…
```

A model reads a fixed amount of text at once, its **context window**, so retrieval selects the few fragments most worth that budget rather than handing over the whole document. The block is the last thing Quiver produces — a plain string, ready to hand to whatever model the app runs.

> Experiment: **The Quiver Notebook** is the right place to watch retrieval narrow a document to its relevant fragments. Load `Dataset.glove50d`, chunk a short multi-paragraph passage, and retrieve the top fragments for a question — then change the question and re-run. The same chunks re-rank, a different block assembles, and the fragment that answers the new question rises to the top. That shift is retrieval doing its one job. See <doc:Quiver-Notebook>.
