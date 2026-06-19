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

// MARK: - Chunk

/// A single fragment of a larger document, paired with its position.
///
/// A `Chunk` keeps the `index` it was cut at, so a fragment retrieved later
/// stays attributable to where it came from in the source. The `text` is the
/// fragment itself — a paragraph, a sentence, or whatever unit the chunker
/// chose. Because `Chunk` is `Sendable`, it can cross the task boundaries a
/// background ingestion worker introduces, and because it is `Codable`, an index
/// of chunks persists to disk and loads at the next launch.
public struct Chunk: Codable, Sendable, Equatable {

    /// The position of this fragment in the sequence the chunker produced,
    /// starting at zero. A retrieved chunk reports this so the answer can cite
    /// which fragment it came from.
    public let index: Int

    /// The fragment text.
    public let text: String

    /// Creates a chunk at a given position.
    ///
    /// - Parameters:
    ///   - index: The fragment's position in the chunked sequence, starting at zero.
    ///   - text: The fragment text.
    public init(index: Int, text: String) {
        self.index = index
        self.text = text
    }
}

// MARK: - Chunker Protocol

/// A strategy for splitting a document into retrievable fragments.
///
/// A type conforming to `Chunker` turns one string into an ordered array of
/// ``Chunk`` values — paragraphs, sentences, fixed windows, or any other unit.
/// Quiver defines the contract but ships no chunker of its own: where a document
/// is cut is a design decision that depends on the content, while everything
/// downstream — embedding, ranking, reporting — stays the same.
///
/// The chunks a `Chunker` produces flow straight into the rest of the retrieval
/// surface. Each chunk's `text` is embedded by an ``Embedder``, and its `index`
/// rides along so a retrieved fragment can be traced back to its place in the
/// source.
///
/// ## Conforming to Chunker
///
/// Conformance is a single method, `chunk(_:)`. Most strategies cut the document
/// into raw pieces and then trim, drop the empties, and number what remains.
/// That second half is the same for every chunker, so Quiver provides it as
/// ``Swift/Sequence/asChunks()``: a conformer cuts the string and hands the
/// pieces to that helper.
///
/// A paragraph chunker is the simplest conformance — it splits on blank lines
/// and lets the helper do the bookkeeping:
///
/// ```swift
/// import Quiver
///
/// struct ParagraphChunker: Chunker {
///     func chunk(_ text: String) -> [Chunk] {
///         text.components(separatedBy: "\n\n").asChunks()
///     }
/// }
/// ```
///
/// We write the type; Quiver ships the contract, not the chunker. A document
/// that yields nothing — empty input, or only whitespace — returns an empty
/// array, not `nil`: an empty document is a valid result, not a failure.
///
/// ## Swapping one strategy for another
///
/// Because the contract is just "text in, `[Chunk]` out," the same downstream
/// code serves every strategy. Moving from paragraph splitting to sentence
/// splitting changes one line — the chunker passed in — and leaves every line
/// that embeds, ranks, and reports exactly as written. It is the same swap an
/// ``Embedder`` makes one step later in the pipeline.
///
/// The ``Swift/Sequence/asChunks()`` helper is a convenience, not a requirement.
/// A strategy that overlaps its fragments or numbers them differently writes its
/// own loop and conforms just as well — the contract asks only for `[Chunk]`,
/// however they are built.
public protocol Chunker: Sendable {

    /// Splits a document into ordered fragments.
    ///
    /// This is the one requirement of `Chunker` — the method a conforming type
    /// implements, not a method called on an existing value. Implementing it is
    /// what turns a custom type into a chunking strategy the rest of the
    /// retrieval surface accepts.
    ///
    /// Return the fragments in document order, each carrying the `index` of its
    /// position in that order. Return an empty array when the document yields no
    /// fragments — an empty or whitespace-only input. Because the protocol is
    /// `Sendable`, a conforming type must be safe to share across task
    /// boundaries, which matters the moment chunking runs off the main thread.
    ///
    /// Many conformers build their result by cutting the text into pieces and
    /// passing those to ``Swift/Sequence/asChunks()``, which trims, drops empties,
    /// and numbers what remains. Using it is optional: a strategy with its own
    /// indexing implements this method directly.
    ///
    /// - Parameter text: The document to split.
    /// - Returns: The fragments, in document order, or an empty array when the
    ///   document yields none.
    func chunk(_ text: String) -> [Chunk]
}

// MARK: - Chunking a Document

extension String {

    /// Splits this document into fragments using the given strategy.
    ///
    /// Hands the string to the chunker and returns the fragments it produces.
    /// Pairing the split with a `some Chunker` value is what lets the strategy
    /// change without touching the call site: the line that ingests a document
    /// is the same whether it splits on paragraphs, sentences, or sections.
    ///
    /// ```swift
    /// import Quiver
    ///
    /// let passage = "First paragraph.\n\nSecond paragraph."
    /// let chunks = passage.chunked(using: ParagraphChunker())
    /// // [Chunk(index: 0, text: "First paragraph."),
    /// //  Chunk(index: 1, text: "Second paragraph.")]
    /// ```
    ///
    /// - Parameter chunker: The strategy that decides where the document is cut.
    /// - Returns: The fragments the chunker produced, in document order.
    public func chunked(using chunker: some Chunker) -> [Chunk] {
        chunker.chunk(self)
    }
}

// MARK: - Assembling Chunks from Pieces

extension Sequence where Element == String {

    /// Turns a sequence of raw pieces into trimmed, numbered chunks.
    ///
    /// Each piece is trimmed of surrounding whitespace and newlines; pieces that
    /// are empty after trimming are dropped. The pieces that remain are numbered
    /// in order, starting at zero, so the result is the indexed ``Chunk`` array
    /// the retrieval surface expects.
    ///
    /// This is the bookkeeping common to most chunkers, offered so a conformer
    /// can cut the document and leave the trimming, empty-dropping, and indexing
    /// to one call:
    ///
    /// ```swift
    /// import Quiver
    ///
    /// struct ParagraphChunker: Chunker {
    ///     func chunk(_ text: String) -> [Chunk] {
    ///         text.components(separatedBy: "\n\n").asChunks()
    ///     }
    /// }
    /// ```
    ///
    /// It is a convenience, not part of the ``Chunker`` contract. A strategy that
    /// overlaps fragments or numbers them differently can ignore it and build its
    /// `[Chunk]` directly.
    ///
    /// - Returns: The trimmed, non-empty pieces as chunks numbered from zero.
    /// - Complexity: O(*n*) in the number of pieces.
    public func asChunks() -> [Chunk] {
        var chunks: [Chunk] = []
        var index = 0
        for piece in self {
            let trimmed = piece.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            chunks.append(Chunk(index: index, text: trimmed))
            index += 1
        }
        return chunks
    }
}
