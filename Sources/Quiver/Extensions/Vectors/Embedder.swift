// Copyright 2025 Wayne W Bishop. All rights reserved.
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

// MARK: - Embedder Protocol

/// A source of text embeddings.
///
/// A type conforming to `Embedder` turns a string into a fixed-dimension vector
/// of `Double` values, or returns `nil` when the text cannot be embedded — an
/// empty string, or one with no recognized tokens. Quiver defines the contract
/// but ships no embedder of its own: where the vectors come from is ours to
/// decide, while everything downstream stays the same.
///
/// The vector an `Embedder` produces is a plain `[Double]`, so it flows directly
/// into the rest of Quiver's similarity surface — ``Swift/Array/cosineSimilarities(to:)->[Double]``
/// and ``Swift/Array/topIndices(k:labels:)`` — without conversion.
///
/// ## Conforming to Embedder
///
/// Conformance is a single method, `embed(_:)`. A type that implements it gains
/// the whole similarity surface for free: ``Swift/Array/embedded(using:)`` to
/// embed a collection of strings, and ``Swift/Array/mostSimilar(to:k:)`` to rank
/// them against a query. Nothing downstream knows or cares which embedder it was
/// handed.
///
/// A word-vector embedder is the simplest conformance. It averages the vectors
/// of the words it recognizes, reusing ``Swift/Array/embed(using:)`` for the
/// per-word lookup:
///
/// ```swift
/// import Quiver
///
/// struct TableEmbedder: Embedder {
///     let table: [String: [Double]]
///
///     func embed(_ text: String) -> [Double]? {
///         let words = text.tokenize()
///         let vectors = words.embed(using: table)
///         return vectors.meanVector()
///     }
/// }
/// ```
///
/// We write the type; Quiver ships the contract, not the embedder. The `nil`
/// return falls out naturally — no recognized words means no vectors, and
/// ``Swift/Array/meanVector()->[Double]?`` returns `nil` for an empty array.
///
/// Averaging token vectors is deliberately coarse. Because it sums and divides,
/// it ignores word order entirely — "a long slow rise" and "a slow long rise"
/// collapse to the same vector. It is the right starting point for learning the
/// pipeline and a serviceable baseline; a richer source that encodes order and
/// context conforms through the very same method when we need it.
///
/// ## Swapping one source for another
///
/// Because the contract is just "text in, `[Double]?` out," the same downstream
/// code serves every source. Moving from a small word-vector table to a full
/// on-device sentence model changes one line — the embedder passed in — and
/// leaves every line that tokenizes, ranks, and reports exactly as written.
public protocol Embedder: Sendable {
    /// Embeds a single piece of text.
    ///
    /// This is the one requirement of `Embedder` — the method a conforming type
    /// implements, not a method called on an existing value. Implementing it is
    /// what turns a custom type into an embedding source the rest of the
    /// similarity surface accepts.
    ///
    /// It is distinct from ``Swift/Array/embed(using:)``, which is a separate
    /// Quiver method on `[String]` that looks each token up in a `[String: [Double]]`
    /// table. A table-backed conformance often calls that array method inside this
    /// one — tokenize, look up each word, average — but the two are different
    /// operations: `embed(_:)` takes one `String` and returns one vector for the
    /// whole text, while `embed(using:)` takes many tokens and returns one vector
    /// per recognized token.
    ///
    /// Return `nil` whenever no vector can be produced — an empty string, or text
    /// with no recognized tokens. Callers rely on that signal, so an implementation
    /// must not substitute a zero vector or a default in its place. Because the
    /// protocol is `Sendable`, a conforming type must be safe to share across task
    /// boundaries, which matters the moment an embedder runs its work off the main
    /// thread.
    ///
    /// - Parameter text: The text to embed.
    /// - Returns: A fixed-dimension vector, or `nil` when the text cannot be
    ///   embedded (empty input, or no recognized tokens).
    func embed(_ text: String) -> [Double]?
}

// MARK: - Batch Embedding

extension Array where Element == String {

    /// Embeds every string in this array, pairing each vector with its source text.
    ///
    /// Each string is passed to the embedder in turn. Strings the embedder
    /// returns `nil` for — empty text, or text with no recognized tokens — are
    /// omitted from the result, so the returned array may be shorter than `self`.
    /// Each surviving vector stays paired with the text it came from, so a
    /// dropped string can never shift the labels of the strings that follow it.
    ///
    /// ```swift
    /// import Quiver
    ///
    /// let docs = ["a long slow rise", "knead the dough well", "proof the yeast"]
    /// let embedded = docs.embedded(using: embedder)
    ///
    /// if let query = embedder.embed("how long should bread rise") {
    ///     let hits = embedded.mostSimilar(to: query, k: 3)
    ///     print(hits.first?.text ?? "no match")
    /// }
    /// ```
    ///
    /// - Parameter embedder: The source of embeddings to apply to each string.
    /// - Returns: A `(text, vector)` pair for each successfully embedded string, in order.
    /// - Complexity: O(*n*) calls to the embedder, where *n* is the number of strings.
    public func embedded(using embedder: some Embedder) -> [(text: String, vector: [Double])] {
        compactMap { text in
            embedder.embed(text).map { (text, $0) }
        }
    }
}

// MARK: - Ranking Embedded Text

extension Array {

    /// Ranks these embedded texts by cosine similarity to a query vector.
    ///
    /// Call this on the output of ``Swift/Array/embedded(using:)``. Because each
    /// text stays paired with its own vector, the returned labels are always the
    /// texts that actually produced the scores — there is no parallel array to
    /// keep in step.
    ///
    /// ```swift
    /// import Quiver
    ///
    /// let docs = ["a long slow rise", "knead the dough well", "proof the yeast"]
    /// let embedded = docs.embedded(using: embedder)
    ///
    /// if let query = embedder.embed("how long should bread rise") {
    ///     let hits = embedded.mostSimilar(to: query, k: 3)
    ///     print(hits.first?.text ?? "no match")
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - query: The query vector to compare each embedded text against.
    ///   - k: The number of top matches to return.
    /// - Returns: The top `k` matches as `(rank, text, score)`, highest score first.
    /// - Complexity: O(*n* log *n*) where *n* is the number of embedded texts.
    public func mostSimilar<Label>(
        to query: [Double],
        k: Int
    ) -> [(rank: Int, text: Label, score: Double)]
    where Element == (text: Label, vector: [Double]) {
        let vectors = map(\.vector)
        let texts = map(\.text)
        return vectors.cosineSimilarities(to: query)
            .topIndices(k: k, labels: texts)
            .map { (rank: $0.rank, text: $0.label, score: $0.score) }
    }
}
