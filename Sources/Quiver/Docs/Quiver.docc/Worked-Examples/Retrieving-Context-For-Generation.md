# Retrieving Context for Generation

Assembling retrieved context blocks from chunked documents to ground a language model's answer.

## Overview

Retrieval is often discussed only as a way to help language models. The math behind it serves many practical purposes beyond generating text. Finding a specific product in a large catalog or a citation in a long report uses the exact same ranking logic. We call this broader pattern **retrieval-augmented context** (commonly known as RAG).

The core operation is a choice of discovery. We use numeric vectors to identify the most relevant fragments from a larger dataset. This mathematical search uncovers relationships that keyword matching would miss entirely. For example, a search for a "fast shoe" can surface a "racing flat" because the two concepts occupy a similar position in vector space. The math understands the relationship even when the words do not match.

## Many applications

The "generation" part is simply one possible outcome for the data we find. A language model can turn those fragments into a natural sentence, and a search engine can present them as a ranked list of products. We can even use this same logic to build automated summaries or to group related documents together.

Quiver provides the mathematical foundation for this work. It handles the search itself so we can focus on how to present the results. This makes advanced discovery tools approachable for any app that needs to find meaning in its data.

> Note: This article builds on <doc:Semantic-Search>, which teaches the embedding-and-ranking pipeline, and <doc:Embedding-Sources>, which defines the swappable source of vectors. The examples here are self-contained, but the vocabulary of vectors and similarity carries over. The embedding values shown are illustrative, chosen so the ranking is easy to follow.

## Splitting documents into chunks

A whole document is too large to retrieve against directly, so the first step is to split each one into **chunks**: fragments small enough to retrieve precise information and large enough to carry a coherent thought.

Quiver ships the ``Chunker`` protocol but no chunker of its own because chunking is a design decision. Where to cut, how large to make each piece, and how much neighboring chunks should overlap all depend on the documents being indexed. We conform a type to `Chunker` and Quiver's `chunked(using:)` calls it, the same arrangement an `Embedder` uses one step later. A paragraph-aware splitter is a good starting point: it cuts on the document's own structure rather than splitting sentences at arbitrary points.

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

We embed through an `Embedder` source (the object <doc:Embedding-Sources> defines) so the vectors can come from a hand-built table while learning the pipeline and from an on-device sentence model in production, with no change to the code around them. The vectors are held in an `EmbeddingIndex`: the value that stores each chunk's embedding beside the chunk and ranks them against a query.

```swift
import Quiver

// Our own label type, not one Quiver requires — the index stores it generically
// beside each vector. It carries provenance to the model: `sourceID` names the
// document, `chunk.index` names the fragment, and `citedForm` renders both as a
// tag, e.g. "[bread-guide#0] Let the dough…". The index owns the vector, so this
// type does not carry one.
struct SourcedChunk: Codable, Equatable {
    let chunk: Chunk
    let sourceID: String

    var citedForm: String { "[\(sourceID)#\(chunk.index)] \(chunk.text)" }
}

// The index takes the embedder once and vectorizes each chunk's text on the way
// in. `add` embeds the text and stores it beside its label in one step — no
// parallel vectors array to keep aligned, and an unembeddable chunk is skipped
// rather than half-stored.
var index = EmbeddingIndex<SourcedChunk>(embedder: embedder)
for chunk in document.chunked(using: ParagraphChunker()) {
    index.add(chunk.text, label: SourcedChunk(chunk: chunk, sourceID: "bread-guide"))
}
```

The label is whatever the app wants each match to carry, here a `SourcedChunk` holding the chunk and its document tag. Making the label `Codable` is what lets the index persist to disk and load at the next launch, so a document is embedded once in its lifetime and queried for free thereafter. That works only because `Chunk` is itself `Codable`. The index holds the `(text, vector)` pairing <doc:Embedding-Sources> establishes, now carrying provenance too: an index that stored only the searchable vector would have to reconstruct the readable text on the way out, but pairing the text with its vector means there is nothing to reconstruct. What we rank is what we read.

## Retrieving the relevant chunks

With the index built, retrieval is a single call. Only the query is embedded at search time. The chunks were embedded at ingest. `retrieve` scores every stored chunk by cosine similarity and returns the closest few, each hit already holding the `SourcedChunk` that produced its score, so provenance survives the ranking without a second lookup:

```swift
let result = index.retrieve("how long should the dough rise", k: 2)

for hit in result.hits {
    print(hit.rank, hit.score, hit.label.chunk.text)
}
// rank 1   0.9939   Let the dough rise slowly. A slow proof develops flavor…
// rank 2   0.8046   Knead the dough until smooth. Good kneading builds…
```

Each `RetrievedHit` carries its `SourcedChunk` as the label, so text, position, and source survive the ranking with no array to index back into. The raw cosine value arrives on `hit.score` untouched. It is the correct measure to retain rather than a percentage. Because it represents similarity, a retrieval system uses it as a threshold to determine relevance, not as a confidence score that the answer is correct.

The result keeps more than the ranked few. `result.scores` is every chunk's similarity, in storage order, and `result.mean`, `result.standardDeviation`, and `result.topZScore` describe the field as a whole, so the geometry behind a ranking stays inspectable, not just the answer it produced.

## Deciding what counts as relevant

A ranking is not yet a decision. The top match always exists, even for a question the corpus does not cover. It is simply the least-bad of a weak field. Turning a ranking into "worth answering" is a judgment, and `isAboveGate(floor:outlierZ:)` makes it once the caller supplies the bar:

```swift
if result.isAboveGate(floor: 0.30, outlierZ: 3.0) {
    // the corpus covers this question — assemble a context block
} else {
    // nothing stands out — the corpus does not cover this question
}
```

The top match passes when its score clears the floor on its own, or when it stands out as an outlier above the field by the given number of standard deviations, and either path is enough, which keeps the gate permissive. The thresholds are required, with no defaults, on purpose: a cosine value carries no fixed meaning across embedders, so a cutoff that signals a strong match for one source is unremarkable for another. The gate supplies the composition; the cutoff stays with the caller, who alone knows the embedder and the corpus.

The outlier path needs a field with enough spread to measure. When a corpus holds only one or two chunks, the standard deviation is undefined or near zero, so the z-score cannot be computed and the gate falls back entirely to the `floor`. A small corpus is decided on the absolute score alone, which is the safe behavior. The gate never errors on a thin field; it simply leans on the bar that always applies.

## Assembling the context block

The retrieved chunks become a single **context block**: the formatted text handed to the model. Assembly is plain string work: walk the hits and join each chunk's `citedForm` (its text tagged with provenance) so the model can attribute what it reads. Because the citation format lives on the label, the loop never has to reconstruct the string. The block is bounded by `k` and by chunk size, the two ends of the chunking tradeoff, so it stays inside the model's input limit without a separate budget:

```swift
// Build the context block from the ranked hits — each carries its own label.
// The separator goes between entries, not after each, so the block carries no
// trailing blank line.
var block = ""
for hit in result.hits {
    let separator = block.isEmpty ? "" : "\n\n"
    block += separator + hit.label.citedForm
}
// [bread-guide#0] Let the dough rise slowly. A slow proof develops flavor…
//
// [bread-guide#1] Knead the dough until smooth. Good kneading builds…
```

A model reads a fixed amount of text at once, its **context window**, so retrieval selects the few fragments most worth that budget rather than handing over the whole document. The block is the last thing Quiver produces: a plain string, ready to hand to whatever model the app runs.

Where the pipeline ends is the point. Quiver returns a `String` and stops, binding the retrieval half to no particular generator. The same block feeds an on-device model on a phone, a hosted model behind an API, or a server-side model on Linux, with no change to the retrieval that produced it. The generator is the caller's choice, made after retrieval has done its work, which is why the same pipeline serves an app and a server without a fork in the code.

## Persisting the index

A document is embedded once and queried many times, so the index is worth keeping between launches. An index persists through the same encode-and-decode pattern a trained model uses: its `snapshot` is a `Codable` value carrying the entries, and `JSONEncoder` writes it to disk.

```swift
// Save: encode the snapshot, the same call a model uses.
let data = try JSONEncoder().encode(index.snapshot)
try data.write(to: url)

// Restore: decode the snapshot, then pair it with an embedder.
let saved = try Data(contentsOf: url)
let snapshot = try JSONDecoder().decode(
    EmbeddingIndex<SourcedChunk>.Snapshot.self, from: saved)
let index = EmbeddingIndex(snapshot, embedder: embedder)
```

The embedder is not part of the snapshot. It is a model or a table, not data, so it is supplied again at reconstruction rather than serialized. What persists is the field of vectors and their labels, the part that was expensive to compute, so a document is embedded once in its lifetime and searched for free thereafter. See <doc:Model-Persistence> for the encode-and-decode shape this shares with every persisted value.

> Important: The saved vectors belong to the embedder that produced them. Cosine similarity is only meaningful when the query and the stored chunks occupy the same vector space, so switching to an embedder of a different architecture or dimension invalidates a saved index. After such a change, discard the saved file and re-embed the corpus.

> Experiment: **The Quiver Notebook** is the right place to watch retrieval narrow a document to its relevant fragments. Load `Dataset.glove50d`, chunk a short multi-paragraph passage, and retrieve the top fragments for a question, then change the question and re-run. The same chunks re-rank, a different block assembles, and `result.topZScore` rises as the answering fragment pulls ahead of the field. That shift is retrieval doing its one job. See <doc:Quiver-Notebook>.

## See Also

- <doc:Semantic-Search>
- <doc:Embedding-Sources>
- <doc:Model-Persistence>
