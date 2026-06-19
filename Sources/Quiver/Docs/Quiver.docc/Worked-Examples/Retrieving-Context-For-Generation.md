# Retrieving Context for Generation

Assembling retrieved context blocks from chunked documents to ground a language model's answer.

## Overview

Retrieval is often discussed only as a way to help language models. However the math behind it serves many practical purposes beyond generating text. Finding a specific product in a large catalog or a citation in a long report uses the exact same ranking logic. We call this broader pattern **retrieval-augmented context** (commonly known as RAG).

The core operation is a choice of discovery. We use numeric vectors to identify the most relevant fragments from a larger dataset. This mathematical search uncovers
relationships that keyword matching would miss entirely. For example a search for a "fast shoe" can surface a "racing flat" because the two concepts occupy a similar position in vector space. The math understands the relationship even if the words do not match.


## Many applications

The "generation" part is simply one possible outcome for the data we find. A language model can turn those fragments into a natural sentence but a search engine can
 present them as a ranked list of products. We can even use this same logic to build automated summaries or to group related documents together. 

 Quiver provides the mathematical foundation for this work. It handles the heavy lifting of the search so we can focus on how to present the results. This makes advanced discovery tools approachable for any app that needs to find meaning in its data.

> Note: This article builds on <doc:Semantic-Search>, which teaches the embedding-and-ranking pipeline, and <doc:Embedding-Sources>, which defines the swappable source of vectors. The examples here are self-contained, but the vocabulary of vectors and similarity carries over. The embedding values shown are illustrative, chosen so the ranking is easy to follow.

## Splitting documents into chunks

A whole document is too large to retrieve against directly, so the first step is to split each one into **chunks**: fragments small enough to retrieve precise information and large enough to carry a coherent thought.

Quiver ships the ``Chunker`` protocol but no chunker of its own because chunking is a design decision. Where to cut, how large to make each piece, and how much neighboring chunks should overlap all depend on the documents being indexed. We conform a type to `Chunker` and Quiver's `chunked(using:)` calls it — the same arrangement an `Embedder` uses one step later. A paragraph-aware splitter is a good starting point: it cuts on the document's own structure rather than splitting sentences at arbitrary points.

```swift
import Quiver

// A Chunker that splits a document on paragraph boundaries. Quiver ships the
// `Chunker` contract and the `Chunk` type (index + text); we supply the
// strategy. The cut is the only thing that varies, so the conformer hands its
// pieces to `asChunks()`, which trims them, drops the empties, and numbers what
// remains — the bookkeeping every chunker would otherwise repeat.
struct ParagraphChunker: Chunker {
    func chunk(_ text: String) -> [Chunk] {
        text.components(separatedBy: "\n\n").asChunks()
    }
}
```

A `Chunk` carries its `index` so a retrieved fragment stays attributable to where it sat in the document. That position is half of provenance; the other half — which document it came from — is added at ingest, when the chunk is paired with its vector and its source. <doc:Embedding-Sources> establishes the `(text, vector)` pairing; the index built next extends it with the source tag that survives all the way to the final block.

Paragraph splitting is the baseline, not the ceiling. Two refinements matter in practice and are worth naming. Small chunks retrieve precisely but fragment context; large chunks keep context coherent but dilute the similarity score and waste the model's limited input budget. And a fixed split can strand an answer that straddles a boundary, so production splitters carry a small overlap from one chunk into the next. Both are tuning decisions for the developer's own documents; the math downstream does not change when the split does.

## Embedding the chunks once

Retrieval is a write-once, read-many pattern. The chunks are embedded a single time when a document enters the system, the vectors are stored, and only the query is embedded at search time. Embedding every chunk on every query would pay the expensive step repeatedly for a result that never changes.

We embed through an `Embedder` source (the object <doc:Embedding-Sources> defines) so the vectors can come from a hand-built table while learning the pipeline and from an on-device sentence model in production, with no change to the code around them. Each chunk's vector is stored alongside the chunk itself, so provenance rides along into the index:

```swift
import Quiver

// Our own type, not one Quiver requires — the ranking methods take it as a
// label generically. Binding chunk and vector in one value keeps them
// inseparable (no two hand-aligned arrays to desync), and `sourceID` +
// `citedForm` carry provenance to the model, e.g. "[bread-guide#0] Let the
// dough…". Codable lets the index persist; Sendable lets it build off the main
// thread. Both are for our app, not for Quiver.
struct StoredChunk: Codable, Sendable {
    let chunk: Chunk
    let vector: [Double]
    let sourceID: String

    var citedForm: String { "[\(sourceID)#\(chunk.index)] \(chunk.text)" }
}

// `embedder` is any `some Embedder`; the chunks come from a `some Chunker`.
// `chunked(using:)` is Quiver's method — it hands the document to the chunker.
let chunks = document.chunked(using: ParagraphChunker())

// Embed each chunk once, at ingest. Chunk, vector, and source go into `stored`
// together — never as separate arrays the code has to keep aligned itself.
var stored: [StoredChunk] = []
for chunk in chunks {
    if let vector = embedder.embed(chunk.text) {
        stored.append(StoredChunk(chunk: chunk, vector: vector, sourceID: "bread-guide"))
    }
}
```

Each `StoredChunk` is our own type, not one Quiver requires: the ranking methods accept it as a label generically, so the pipeline works whatever shape we choose. The two conformances are for our app, not for Quiver. `Codable` is what lets the index persist to disk and load at the next launch, so a document is embedded once in its lifetime and queried for free thereafter. That works only because `Chunk` is itself `Codable`. `Sendable` lets the index be built off the main thread, in a background task or a server request handler, without crossing an isolation boundary unsafely. Drop the persistence and the concurrency and both conformances fall away; keep them and the index is a durable, shareable value. See <doc:Model-Persistence> for the encode-and-decode shape this index shares with every persisted value.

Pairing the chunk with its vector is the same `(text, vector)` honesty <doc:Embedding-Sources> establishes, now carrying provenance too. An index that stored only the searchable vector would have to reconstruct the readable text on the way out; pairing the text with its vector means there is nothing to reconstruct — what we rank is what we read.

## Retrieving the relevant chunks

With the index built, retrieval is the ranking pipeline <doc:Semantic-Search> teaches, run against the stored vectors. The query is embedded through the same source, then `cosineSimilarities(to:)` scores every chunk in one call. `topIndices(k:labels:)` reads off the closest few, carrying each stored chunk through as the label, so provenance survives the ranking without a second lookup:

```swift
import Quiver

// Only the query is embedded at search time — the chunks were embedded at ingest.
guard let queryVector = embedder.embed("how long should the dough rise") else {
    return
}

// `stored` is the single source of truth. Derive the parallel views the ranking
// needs — vectors to score, stored chunks to carry through as labels — both in
// `stored` order.
var vectors: [[Double]] = []
for item in stored {
    vectors.append(item.vector)
}

let scores = vectors.cosineSimilarities(to: queryVector)
let hits = scores.topIndices(k: 2, labels: stored)
// hits[0]: (rank: 1, label: StoredChunk(bread-guide#0, …), score: 0.9939)
// hits[1]: (rank: 2, label: StoredChunk(bread-guide#1, …), score: 0.8046)
```

Passing the stored chunks as `labels` is what keeps retrieval honest: each hit arrives already holding the `StoredChunk` that produced its score (text, position, and source) so there is no array to index back into after ranking. The vectors are derived from `stored` for this one call and discarded; `stored` stays the single source of truth, the pairing intact. The raw cosine value is the correct measure to retain here rather than a percentage. Because it represents similarity, a retrieval system uses it as a threshold to determine relevance, not as a confidence score that the answer is correct.

## Assembling the context block

The retrieved chunks become a single **context block**: the formatted text handed to the model. Assembly is plain string work: walk the hits and join each stored chunk's `citedForm` (its text tagged with provenance) so the model can attribute what it reads. Because the citation format lives on `StoredChunk`, the loop never has to reconstruct the string. The block is bounded by `k` and by chunk size, the two ends of the chunking tradeoff, so it stays inside the model's input limit without a separate budget:

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

A model reads a fixed amount of text at once, its **context window**, so retrieval selects the few fragments most worth that budget rather than handing over the whole document. The block is the last thing Quiver produces: a plain string, ready to hand to whatever model the app runs.

> Experiment: **The Quiver Notebook** is the right place to watch retrieval narrow a document to its relevant fragments. Load `Dataset.glove50d`, chunk a short multi-paragraph passage, and retrieve the top fragments for a question — then change the question and re-run. The same chunks re-rank, a different block assembles, and the fragment that answers the new question rises to the top. That shift is retrieval doing its one job. See <doc:Quiver-Notebook>.
