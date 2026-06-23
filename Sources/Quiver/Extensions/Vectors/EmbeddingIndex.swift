// Copyright 2026 Wayne W Bishop. All rights reserved.
//
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under
// the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
// ANY KIND, either express or implied. See the License for the specific language governing
// permissions and limitations under the License.

import Foundation

// MARK: - RetrievedHit

/// One ranked match from a retrieval: a stored payload paired with its
/// similarity to the query.
///
/// A `RetrievedHit` is the labeled, named form of the `(rank, label, score)`
/// tuple that ``Swift/Array/topIndices(k:labels:)`` produces. The `score` is the
/// raw cosine similarity — a value in `-1...1`, higher meaning closer in
/// meaning — carried through untouched so a caller can read, log, or threshold
/// it. `Label` is whatever payload was stored: a `String`, a ``Chunk``, or a
/// custom source-tagged type.
public struct RetrievedHit<Label: Codable & Equatable>: Codable, Equatable, CustomStringConvertible {

    /// The match's position in the ranked results, starting at one.
    public let rank: Int

    /// The stored payload this hit refers to.
    public let label: Label

    /// The cosine similarity of this hit to the query, in `-1...1`.
    public let score: Double

    /// Creates a ranked hit.
    public init(rank: Int, label: Label, score: Double) {
        self.rank = rank
        self.label = label
        self.score = score
    }

    public var description: String {
        "\(rank). \(label)  (\(String(format: "%.4f", score)))"
    }
}

// MARK: - RetrievalResult

/// The outcome of one retrieval: the top-k hits, plus the full score field they
/// were drawn from and its summary statistics.
///
/// Retrieval computes a similarity score for *every* stored entry, then returns
/// the closest `k`. `RetrievalResult` keeps both halves: the ranked ``hits`` for
/// the common case, and the full ``scores`` field with its ``mean``,
/// ``standardDeviation``, and ``topZScore`` so the math behind the ranking stays
/// visible. Nothing is hidden — a caller can inspect the whole distribution,
/// apply their own relevance rule, or print the result to read it at a glance.
///
/// The result describes and ranks; it does not judge relevance. Whether the top
/// hit is *good enough* to act on is the caller's decision (see ``isAboveGate(floor:outlierZ:)``),
/// and whether the retrieved passages actually answer a question is a downstream
/// language model's job.
public struct RetrievalResult<Label: Codable & Equatable>: CustomStringConvertible {

    /// The top-k matches, ranked best first.
    public let hits: [RetrievedHit<Label>]

    /// The cosine score of every stored entry against the query, in storage
    /// order — the full field, not just the returned hits.
    public let scores: [Double]

    /// The mean of the full score field.
    public let mean: Double

    /// The standard deviation (sample, `ddof: 1`) of the full score field.
    public let standardDeviation: Double

    /// The highest score in the field — the top hit's score.
    public let topScore: Double

    /// How many standard deviations the top hit sits above the field's mean.
    /// Zero when the field has no spread (every score identical).
    public let topZScore: Double

    /// Creates a retrieval result. Callers normally receive one from
    /// ``EmbeddingIndex/retrieve(_:k:)`` rather than building it directly.
    public init(
        hits: [RetrievedHit<Label>],
        scores: [Double],
        mean: Double,
        standardDeviation: Double,
        topScore: Double,
        topZScore: Double
    ) {
        self.hits = hits
        self.scores = scores
        self.mean = mean
        self.standardDeviation = standardDeviation
        self.topScore = topScore
        self.topZScore = topZScore
    }

    /// Whether the top hit clears a relevance gate the caller defines: its score
    /// is high enough on its own (`floor`), **or** it stands out as a clear
    /// outlier above the field (`outlierZ` standard deviations).
    ///
    /// The thresholds are required, with no defaults, on purpose: the right
    /// values depend on the embedder and the corpus, so this method supplies the
    /// composition — high-enough OR outlier — and never the cutoff. A permissive
    /// OR is the point: it lets anything plausibly relevant through and turns
    /// away only obvious nothing, leaving the final judgment to whatever reads
    /// the retrieved passages.
    ///
    /// - Parameters:
    ///   - floor: The absolute score a top hit may clear on its own.
    ///   - outlierZ: The number of standard deviations above the field a top hit
    ///     may stand out by instead.
    /// - Returns: `true` if the top hit clears either bar; `false` if the result
    ///   is empty or the top hit clears neither.
    public func isAboveGate(floor: Double, outlierZ: Double) -> Bool {
        guard let best = hits.first else { return false }
        let z = scores.zScore(of: best.score) ?? 0
        return best.score >= floor || z >= outlierZ
    }

    public var description: String {
        guard !hits.isEmpty else {
            return "retrieval: no matches"
        }
        var lines: [String] = []
        for hit in hits {
            lines.append("  rank \(hit.rank)   \(String(format: "%.4f", hit.score))   \(hit.label)")
        }
        lines.append("")
        lines.append("  field:   \(scores.count) scored · mean \(String(format: "%.4f", mean)) · spread \(String(format: "%.4f", standardDeviation))")
        lines.append("  top hit: \(String(format: "%.4f", topScore)) → \(String(format: "%.2f", topZScore))σ above the field")
        lines.append("  note:    raw cosine similarities; the caller's gate and the model decide relevance")
        return lines.joined(separator: "\n")
    }
}

// MARK: - EmbeddingIndex

/// An on-device vector store: embed each entry once at ingest, then rank the
/// whole corpus against a query by cosine similarity.
///
/// `EmbeddingIndex` packages the embed → score → rank pipeline into one value
/// you hold, instead of orchestrating the loose calls by hand. It is generic
/// over the payload `Label` — store `String`, ``Chunk``, or any `Codable` and
/// `Equatable` type — and takes an ``Embedder`` to turn text into vectors. The
/// embedder is text-to-vector only; it never sees `Label`. The caller pairs the
/// two at ``add(_:label:)``: the text is embedded, the label is stored alongside
/// its vector.
///
/// The index ranks and exposes its math; it does not judge relevance. Every
/// value behind a ranking stays reachable — the full score field on
/// ``RetrievalResult/scores``, the stored ``vectors``, and the per-entry
/// embedding via ``vector(for:)`` — so the geometry can always be inspected and
/// verified, never hidden behind the convenience.
///
/// ```swift
/// import Quiver
///
/// var index = EmbeddingIndex<Chunk>(embedder: myEmbedder)
/// for chunk in document.chunked(using: ParagraphChunker()) {
///     index.add(chunk.text, label: chunk)
/// }
///
/// let result = index.retrieve("how long should the dough rise", k: 3)
/// for hit in result.hits {
///     print(hit.rank, hit.score, hit.label.text)
/// }
/// ```
///
/// This is an exact, in-memory store — it scores the query against every entry,
/// which is correct and fast for thousands to low-millions of vectors. Past that
/// range the right move is an approximate-nearest-neighbor index, with the same
/// cosine ranking inside it.
public struct EmbeddingIndex<Label: Codable & Equatable>: Sequence, CustomStringConvertible {

    /// One stored entry: a payload and its embedding, in ingest order.
    /// `Codable` so a built index persists to disk and loads at the next launch.
    public struct Entry: Codable, Equatable {
        /// The stored payload.
        public let label: Label
        /// The embedding produced for this entry's text.
        public let vector: [Double]

        public init(label: Label, vector: [Double]) {
            self.label = label
            self.vector = vector
        }
    }

    /// The stored entries, in ingest order.
    private var entries: [Entry]

    /// The embedder used to vectorize text on ingest and on query. Not part of
    /// the persisted state; it is re-supplied when an index is rebuilt from a
    /// ``Snapshot``.
    private let embedder: any Embedder

    /// Creates an empty index that vectorizes text with the given embedder.
    public init(embedder: some Embedder) {
        self.entries = []
        self.embedder = embedder
    }

    // MARK: Ingest

    /// Embeds `text` once and stores it under `label`.
    ///
    /// If the embedder cannot produce a vector for the text (it returns `nil`),
    /// the entry is skipped rather than stored — an unembeddable entry never
    /// enters the index.
    public mutating func add(_ text: String, label: Label) {
        guard let vector = embedder.embed(text) else { return }
        entries.append(Entry(label: label, vector: vector))
    }

    /// Stores a pre-computed `vector` under `label`, without embedding.
    ///
    /// Use this when the embedding was produced elsewhere — loaded from a file,
    /// or computed by a different path. The vector should share the dimension of
    /// every other stored vector.
    public mutating func add(vector: [Double], label: Label) {
        entries.append(Entry(label: label, vector: vector))
    }

    // MARK: Removal

    /// Removes every entry whose label equals `label`.
    public mutating func remove(label: Label) {
        entries.removeAll { $0.label == label }
    }

    /// Removes every entry whose label is contained in `labels` — for dropping a
    /// document's entries before re-adding them, without rebuilding the index.
    public mutating func remove(labels: [Label]) {
        entries.removeAll { labels.contains($0.label) }
    }

    /// Removes all entries, leaving the embedder in place.
    public mutating func removeAll() {
        entries.removeAll()
    }

    // MARK: Retrieve

    /// Embeds the query, scores every entry by cosine similarity, and returns the
    /// top `k` matches along with the full score field and its statistics.
    ///
    /// Returns an empty result when the query cannot be embedded or the index is
    /// empty.
    ///
    /// - Parameters:
    ///   - query: The text to search for.
    ///   - k: The number of top matches to return.
    /// - Returns: A ``RetrievalResult`` carrying the ranked hits plus the full
    ///   field, its mean, spread, and the top hit's z-score.
    public func retrieve(_ query: String, k: Int) -> RetrievalResult<Label> {
        guard !entries.isEmpty, let queryVector = embedder.embed(query) else {
            return RetrievalResult(hits: [], scores: [], mean: 0, standardDeviation: 0, topScore: 0, topZScore: 0)
        }

        let labels = entries.map(\.label)
        let scores = entries.map(\.vector).cosineSimilarities(to: queryVector)

        let hits = scores.topIndices(k: k, labels: labels).map { hit in
            RetrievedHit(rank: hit.rank, label: hit.label, score: hit.score)
        }

        let mean = scores.mean() ?? 0
        let sd = scores.standardDeviation() ?? 0
        let top = scores.max() ?? 0
        let topZ = scores.zScore(of: top) ?? 0

        return RetrievalResult(
            hits: hits,
            scores: scores,
            mean: mean,
            standardDeviation: sd,
            topScore: top,
            topZScore: topZ
        )
    }

    // MARK: Inspecting the math

    /// The stored vectors, in ingest order — the corpus matrix. Pair any two to
    /// inspect their similarity directly, e.g.
    /// `index.vectors[i].cosineOfAngle(with: index.vectors[j])`.
    public var vectors: [[Double]] {
        entries.map(\.vector)
    }

    /// The cosine score of every entry against `query`, in storage order, without
    /// ranking — for inspecting the full field or applying a custom relevance
    /// rule. Empty when the query cannot be embedded.
    public func scores(for query: String) -> [Double] {
        guard let queryVector = embedder.embed(query) else { return [] }
        return entries.map(\.vector).cosineSimilarities(to: queryVector)
    }

    /// The stored embedding for the first entry whose label equals `label`, or
    /// `nil` if no such entry exists.
    public func vector(for label: Label) -> [Double]? {
        entries.first { $0.label == label }?.vector
    }

    // MARK: Introspection

    /// The number of stored entries.
    public var count: Int { entries.count }

    /// Whether the index holds no entries.
    public var isEmpty: Bool { entries.isEmpty }

    // MARK: Sequence

    /// Iterates the stored entries in ingest order.
    public func makeIterator() -> IndexingIterator<[Entry]> {
        entries.makeIterator()
    }

    // MARK: Persistence

    /// The persistable state of an index: its entries, labels and vectors,
    /// without the embedder.
    ///
    /// `Snapshot` is the `Codable` value an index encodes to. It follows the
    /// same encode-and-decode pattern as a trained model (see
    /// <doc:Model-Persistence>): encode the snapshot to JSON, write it to disk,
    /// and decode it back. The one difference is the embedder, which is a model
    /// or a table rather than data and so is supplied again at reconstruction
    /// rather than serialized.
    ///
    /// ```swift
    /// // Persist: encode the snapshot, exactly like a model.
    /// let data = try JSONEncoder().encode(index.snapshot)
    ///
    /// // Restore: decode the snapshot, then pair it with an embedder.
    /// let snapshot = try JSONDecoder().decode(
    ///     EmbeddingIndex<SourcedChunk>.Snapshot.self, from: data)
    /// let index = EmbeddingIndex(snapshot, embedder: embedder)
    /// ```
    public struct Snapshot: Codable, Equatable {
        let entries: [Entry]
    }

    /// The index's persistable state — encode this with `JSONEncoder` to save
    /// the embedded corpus to disk, a database, or the network.
    public var snapshot: Snapshot {
        Snapshot(entries: entries)
    }

    /// Reconstructs an index from a decoded ``Snapshot`` and an embedder.
    ///
    /// The snapshot carries the entries, labels and their vectors; the embedder
    /// is supplied here rather than read from the snapshot, so the restored
    /// index can embed new queries and new entries. The vectors belong to the
    /// embedder that produced them, so the embedder passed here must occupy the
    /// same vector space as the one that built the snapshot.
    public init(_ snapshot: Snapshot, embedder: some Embedder) {
        self.entries = snapshot.entries
        self.embedder = embedder
    }

    // MARK: CustomStringConvertible

    public var description: String {
        "EmbeddingIndex(\(count) \(count == 1 ? "entry" : "entries"))"
    }
}
