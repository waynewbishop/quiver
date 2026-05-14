// Copyright 2025 Wayne W Bishop. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import XCTest
@testable import Quiver

final class StringExtensionsTests: XCTestCase {

    // MARK: - Tokenize Tests

    // Covers basic splitting and lowercase conversion
    func testTokenizeBasicText() {
        XCTAssertEqual("Comfortable Running Shoes".tokenize(), ["comfortable", "running", "shoes"])
        XCTAssertEqual("UPPERCASE lowercase MixedCase".tokenize(), ["uppercase", "lowercase", "mixedcase"])
    }

    // Covers multiple spaces, newlines, mixed whitespace, leading/trailing whitespace
    func testTokenizeWhitespaceHandling() {
        XCTAssertEqual("word1    word2     word3".tokenize(), ["word1", "word2", "word3"])
        XCTAssertEqual("line1\nline2\nline3".tokenize(), ["line1", "line2", "line3"])
        XCTAssertEqual("word1  \n  word2\t\tword3\r\nword4".tokenize(),
                        ["word1", "word2", "word3", "word4"])
        XCTAssertEqual("  leading and trailing  ".tokenize(), ["leading", "and", "trailing"])
    }

    // Covers empty string, whitespace-only, single word
    func testTokenizeEdgeCases() {
        XCTAssertEqual("".tokenize(), [])
        XCTAssertEqual("   \n\t  \r\n  ".tokenize(), [])
        XCTAssertEqual("word".tokenize(), ["word"])
    }

    // MARK: - Punctuation removal (default behavior)

    // Covers trailing punctuation, quotes, brackets, hyphens, and punctuation-only tokens
    func testTokenizeRemovesPunctuation() {
        // Mixed sentence punctuation
        XCTAssertEqual("Hello, world! How are you?".tokenize(),
                        ["hello", "world", "how", "are", "you"])

        // Trailing periods
        XCTAssertEqual("end of sentence. next sentence.".tokenize(),
                        ["end", "of", "sentence", "next", "sentence"])

        // Quotes and brackets
        XCTAssertEqual("\"quoted\" [bracketed] (parenthesized)".tokenize(),
                        ["quoted", "bracketed", "parenthesized"])

        // Punctuation-only tokens removed
        XCTAssertEqual("hello ... world !!! done".tokenize(),
                        ["hello", "world", "done"])

        // Leading hyphens removed
        XCTAssertEqual("--flag -option".tokenize(), ["flag", "option"])
    }

    // Apostrophes and other interior punctuation are preserved
    func testTokenizePreservesInteriorPunctuation() {
        XCTAssertEqual("don't can't it's".tokenize(), ["don't", "can't", "it's"])
    }

    // MARK: - Punctuation preservation (opt-in)

    func testTokenizePreservesPunctuationWhenRequested() {
        let text = "Hello, world! How are you?"
        let tokens = text.tokenize(removingPunctuation: false)

        XCTAssertEqual(tokens, ["hello,", "world!", "how", "are", "you?"])
    }

    func testTokenizePreservesPunctuationEdgeCases() {
        XCTAssertEqual("...".tokenize(removingPunctuation: false), ["..."])
        XCTAssertEqual("!!!".tokenize(removingPunctuation: false), ["!!!"])
    }

    // MARK: - Embed Tests

    // Covers basic lookup, unknown-word skipping, and integration with tokenize()
    func testEmbed() {
        let embeddings = [
            "running": [0.8, 0.7, 0.9],
            "shoes": [0.1, 0.9, 0.2]
        ]

        // Basic lookup
        let basic = ["running", "shoes"].embed(using: embeddings)
        XCTAssertEqual(basic.count, 2)
        XCTAssertEqual(basic[0], [0.8, 0.7, 0.9])
        XCTAssertEqual(basic[1], [0.1, 0.9, 0.2])

        // Unknown words are skipped
        let withUnknown = ["running", "unknown", "shoes"].embed(using: embeddings)
        XCTAssertEqual(withUnknown.count, 2)
        XCTAssertEqual(withUnknown[0], [0.8, 0.7, 0.9])
        XCTAssertEqual(withUnknown[1], [0.1, 0.9, 0.2])

        // Tokenize then embed — punctuation must be removed first
        let fromText = "running, shoes!".tokenize().embed(using: embeddings)
        XCTAssertEqual(fromText.count, 2)
        XCTAssertEqual(fromText[0], [0.8, 0.7, 0.9])
        XCTAssertEqual(fromText[1], [0.1, 0.9, 0.2])
    }

    // Covers empty input and no matches
    func testEmbedEdgeCases() {
        let embeddings = [
            "running": [0.8, 0.7, 0.9],
            "shoes": [0.1, 0.9, 0.2]
        ]

        XCTAssertEqual([String]().embed(using: embeddings).count, 0)
        XCTAssertEqual(["unknown1", "unknown2"].embed(using: embeddings).count, 0)
    }
}
