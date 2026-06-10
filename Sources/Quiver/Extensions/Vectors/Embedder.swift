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
/// but ships no embedder of its own: where the vectors come from is yours to
/// decide, while everything downstream stays the same.
///
/// The vector an `Embedder` produces is a plain `[Double]`, so it flows directly
/// into the rest of Quiver's similarity surface — ``Swift/Array/cosineSimilarities(to:)->[Double]``
/// and ``Swift/Array/topIndices(k:labels:)`` — without conversion.
///
/// Conforming is a single method. A bag-of-words embedder over a word-vector
/// table averages the vectors of the words it recognizes:
///
/// ```swift
/// import Quiver
///
/// struct GloVeEmbedder: Embedder {
///     let table: [String: [Double]]
///     func embed(_ text: String) -> [Double]? {
///         text.tokenize().compactMap { table[$0] }.meanVector()
///     }
/// }
/// ```
///
/// Because the contract is just "text in, `[Double]?` out," the same downstream
/// code serves every source. Swapping a classroom word-vector table for an
/// on-device sentence model changes one line — the embedder passed in — and
/// leaves the search pipeline untouched.
public protocol Embedder: Sendable {
    /// Embeds a single piece of text.
    ///
    /// - Parameter text: The text to embed.
    /// - Returns: A fixed-dimension vector, or `nil` when the text cannot be
    ///   embedded (empty input, or no recognized tokens).
    func embed(_ text: String) -> [Double]?
}

// MARK: - Batch Embedding

extension Array where Element == String {

    /// Embeds every string in this array, dropping any the embedder cannot process.
    ///
    /// Each string is passed to the embedder in turn. Strings the embedder
    /// returns `nil` for — empty text, or text with no recognized tokens — are
    /// omitted from the result, so the returned array may be shorter than `self`.
    /// The result is a plain `[[Double]]`, ready for the similarity surface.
    ///
    /// ```swift
    /// import Quiver
    ///
    /// let docs = ["a long slow rise", "knead the dough well", "proof the yeast"]
    /// let vectors = docs.embedded(using: embedder)
    ///
    /// let query = embedder.embed("how long should bread rise")
    /// if let query {
    ///     let hits = vectors.cosineSimilarities(to: query)
    ///                       .topIndices(k: 3, labels: docs)
    ///     print(hits.first?.label ?? "no match")
    /// }
    /// ```
    ///
    /// - Parameter embedder: The source of embeddings to apply to each string.
    /// - Returns: One vector per successfully embedded string, in order.
    public func embedded(using embedder: some Embedder) -> [[Double]] {
        compactMap(embedder.embed)
    }
}
