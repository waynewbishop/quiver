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

import XCTest
import Foundation
@testable import Quiver

final class ChunkerTests: XCTestCase {

    // MARK: - Test conformers

    /// The canonical conformer: cut on blank lines, lean on asChunks() for the
    /// trim/drop/index bookkeeping.
    private struct ParagraphChunker: Chunker {
        func chunk(_ text: String) -> [Chunk] {
            text.components(separatedBy: "\n\n").asChunks()
        }
    }

    /// A conformer that ignores asChunks() entirely and builds its own indexing —
    /// proves the helper is optional and the contract is just `[Chunk]`.
    private struct CharacterOffsetChunker: Chunker {
        func chunk(_ text: String) -> [Chunk] {
            var chunks: [Chunk] = []
            var offset = 0
            for sentence in text.components(separatedBy: ". ") {
                let trimmed = sentence.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    // index is the character offset, not the sequence position.
                    chunks.append(Chunk(index: offset, text: trimmed))
                }
                offset += sentence.count + 2  // account for the ". " delimiter
            }
            return chunks
        }
    }

    // MARK: - Chunk type

    func testChunkEquatable() {
        let a = Chunk(index: 0, text: "hello")
        let b = Chunk(index: 0, text: "hello")
        let c = Chunk(index: 1, text: "hello")
        let d = Chunk(index: 0, text: "world")
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)   // different index
        XCTAssertNotEqual(a, d)   // different text
    }

    func testChunkStoresIndexAndText() {
        let chunk = Chunk(index: 3, text: "fragment")
        XCTAssertEqual(chunk.index, 3)
        XCTAssertEqual(chunk.text, "fragment")
    }

    func testChunkCodableRoundTrip() throws {
        // Chunk is Codable so an index of chunks persists to disk and loads back.
        let original = [
            Chunk(index: 0, text: "first fragment"),
            Chunk(index: 1, text: "second fragment")
        ]
        let data = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode([Chunk].self, from: data)
        XCTAssertEqual(restored, original)
    }

    // MARK: - asChunks() helper

    func testAsChunksNumbersFromZero() {
        let chunks = ["first", "second", "third"].asChunks()
        XCTAssertEqual(chunks, [
            Chunk(index: 0, text: "first"),
            Chunk(index: 1, text: "second"),
            Chunk(index: 2, text: "third")
        ])
    }

    func testAsChunksTrimsWhitespace() {
        let chunks = ["  padded  ", "\n\tlines\n"].asChunks()
        XCTAssertEqual(chunks, [
            Chunk(index: 0, text: "padded"),
            Chunk(index: 1, text: "lines")
        ])
    }

    func testAsChunksDropsEmptyAndReindexes() {
        // The empty/whitespace pieces must drop AND not leave gaps in the index.
        let chunks = ["keep", "   ", "", "alsokeep"].asChunks()
        XCTAssertEqual(chunks, [
            Chunk(index: 0, text: "keep"),
            Chunk(index: 1, text: "alsokeep")   // index 1, not 3 — re-numbered
        ])
    }

    func testAsChunksEmptySequence() {
        let chunks: [String] = []
        XCTAssertEqual(chunks.asChunks(), [])
    }

    func testAsChunksAllWhitespaceYieldsEmpty() {
        let chunks = ["", "   ", "\n\n"].asChunks()
        XCTAssertEqual(chunks, [])
    }

    // MARK: - chunked(using:) calls the conformer

    func testChunkedUsingParagraphChunker() {
        let passage = "First paragraph.\n\nSecond paragraph."
        let chunks = passage.chunked(using: ParagraphChunker())
        XCTAssertEqual(chunks, [
            Chunk(index: 0, text: "First paragraph."),
            Chunk(index: 1, text: "Second paragraph.")
        ])
    }

    func testChunkedDropsBlankParagraphs() {
        // Triple blank lines create an empty middle paragraph that must drop.
        let passage = "One.\n\n\n\nTwo."
        let chunks = passage.chunked(using: ParagraphChunker())
        XCTAssertEqual(chunks, [
            Chunk(index: 0, text: "One."),
            Chunk(index: 1, text: "Two.")
        ])
    }

    func testChunkedEmptyDocumentYieldsEmpty() {
        XCTAssertEqual("".chunked(using: ParagraphChunker()), [])
        XCTAssertEqual("   \n\n   ".chunked(using: ParagraphChunker()), [])
    }

    func testChunkedSingleParagraph() {
        let chunks = "Just one block of text.".chunked(using: ParagraphChunker())
        XCTAssertEqual(chunks, [Chunk(index: 0, text: "Just one block of text.")])
    }

    // MARK: - Custom conformer that ignores asChunks()

    func testCustomConformerWithOwnIndexing() {
        // Proves asChunks() is optional: this chunker numbers by character offset.
        // Splitting on ". " consumes the delimiter, so "Alpha" has no trailing dot
        // but the final "Beta." keeps its period (no delimiter follows it).
        let chunks = "Alpha. Beta.".chunked(using: CharacterOffsetChunker())
        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0].text, "Alpha")
        XCTAssertEqual(chunks[1].text, "Beta.")
        // Indices are offsets, not 0,1 — the contract accepts any [Chunk].
        XCTAssertEqual(chunks[0].index, 0)
        XCTAssertNotEqual(chunks[1].index, 1)   // it's a character offset, > 1
    }

    // MARK: - Swappability (the design payoff)

    func testStrategiesAreSwappable() {
        // Same call site, different strategy — the point of the protocol seam.
        let text = "Sentence one. Sentence two."
        let byParagraph = text.chunked(using: ParagraphChunker())
        let byOffset = text.chunked(using: CharacterOffsetChunker())
        // Paragraph splitting finds one chunk (no blank line); sentence splitting finds two.
        XCTAssertEqual(byParagraph.count, 1)
        XCTAssertEqual(byOffset.count, 2)
    }
}
