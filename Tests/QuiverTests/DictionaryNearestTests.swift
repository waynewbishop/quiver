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

import XCTest
@testable import Quiver

final class DictionaryNearestTests: XCTestCase {

    // MARK: - Happy path

    func testNearestReturnsTopK() {
        let embeddings: [String: [Double]] = [
            "running":  [0.8, 0.7, 0.9, 0.2],
            "jogging":  [0.8, 0.7, 0.8, 0.2],
            "computer": [0.1, 0.3, 0.2, 0.9]
        ]

        let results = embeddings.nearest(to: [0.8, 0.7, 0.9, 0.2], k: 2)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].rank, 1)
        XCTAssertEqual(results[0].word, "running")
        XCTAssertEqual(results[0].score, 1.0, accuracy: 1e-10)
        XCTAssertEqual(results[1].rank, 2)
        XCTAssertEqual(results[1].word, "jogging")
        XCTAssertGreaterThan(results[1].score, 0.99)
    }

    func testNearestRanksByScoreDescending() {
        let embeddings: [String: [Double]] = [
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
            "c": [0.7, 0.7],
            "d": [-1.0, 0.0]
        ]

        let results = embeddings.nearest(to: [1.0, 0.0], k: 4)

        XCTAssertEqual(results.count, 4)
        XCTAssertEqual(results[0].word, "a")
        for i in 0..<results.count - 1 {
            XCTAssertGreaterThanOrEqual(results[i].score, results[i + 1].score)
        }
        XCTAssertEqual(results[3].word, "d")
    }

    func testNearestWithDefaultK() {
        // Default k = 5; building a dict of 7 entries verifies the cap.
        var embeddings: [String: [Double]] = [:]
        for i in 0..<7 {
            embeddings["w\(i)"] = [Double(i), 0.0]
        }

        let results = embeddings.nearest(to: [6.0, 0.0])

        XCTAssertEqual(results.count, 5)
    }

    // MARK: - Analogy demo (king/queen)

    func testAnalogyKingQueen() {
        let embeddings: [String: [Double]] = [
            "king":  [0.9, 0.2, 0.8, 0.7],
            "queen": [0.3, 0.9, 0.8, 0.7],
            "man":   [0.8, 0.1, 0.2, 0.6],
            "woman": [0.2, 0.8, 0.2, 0.6],
            "apple": [0.1, 0.1, 0.1, 0.1]
        ]

        // king - man + woman should land closest to queen.
        let target = embeddings["king"]!
            .subtract(embeddings["man"]!)
            .add(embeddings["woman"]!)

        let results = embeddings.nearest(to: target, k: 1)

        XCTAssertEqual(results.first?.word, "queen")
    }

    // MARK: - Edge cases

    func testNearestOnEmptyDictionary() {
        let embeddings: [String: [Double]] = [:]
        let results = embeddings.nearest(to: [1.0, 0.0, 0.0], k: 3)
        XCTAssertEqual(results.count, 0)
    }

    func testKLargerThanDictionaryReturnsAllEntries() {
        let embeddings: [String: [Double]] = [
            "a": [1.0, 0.0],
            "b": [0.0, 1.0]
        ]

        let results = embeddings.nearest(to: [1.0, 0.0], k: 10)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].rank, 1)
        XCTAssertEqual(results[1].rank, 2)
    }

    func testDimensionMismatchedEntriesAreSkipped() {
        let embeddings: [String: [Double]] = [
            "match_a": [1.0, 0.0, 0.0],
            "wrong_size_50d": Array(repeating: 0.5, count: 50),
            "match_b": [0.5, 0.5, 0.5]
        ]

        let results = embeddings.nearest(to: [1.0, 0.0, 0.0], k: 5)

        XCTAssertEqual(results.count, 2)
        let returnedWords = Set(results.map { $0.word })
        XCTAssertTrue(returnedWords.contains("match_a"))
        XCTAssertTrue(returnedWords.contains("match_b"))
        XCTAssertFalse(returnedWords.contains("wrong_size_50d"))
    }

    func testZeroMagnitudeEntryDoesNotProduceNaN() {
        let embeddings: [String: [Double]] = [
            "real":    [1.0, 0.0],
            "zero":    [0.0, 0.0]
        ]

        let results = embeddings.nearest(to: [1.0, 0.0], k: 2)

        XCTAssertEqual(results.count, 2)
        for result in results {
            XCTAssertFalse(result.score.isNaN, "Zero-magnitude entry produced NaN")
        }
    }

    func testRankIsOneBased() {
        let embeddings: [String: [Double]] = [
            "best":  [1.0, 0.0],
            "mid":   [0.7, 0.7],
            "worst": [-1.0, 0.0]
        ]

        let results = embeddings.nearest(to: [1.0, 0.0], k: 3)

        XCTAssertEqual(results[0].rank, 1)
        XCTAssertEqual(results[1].rank, 2)
        XCTAssertEqual(results[2].rank, 3)
    }

    // MARK: - Full pipeline (tokenize → embed → nearest)

    func testPipelineFromText() {
        let embeddings: [String: [Double]] = [
            "comfortable": [0.7, 0.8, 0.3, 0.1],
            "running":     [0.8, 0.7, 0.9, 0.2],
            "shoes":       [0.6, 0.9, 0.4, 0.1],
            "trail":       [0.4, 0.3, 0.8, 0.7],
            "outdoor":     [0.3, 0.2, 0.7, 0.8]
        ]

        guard let queryVector = "Comfortable Running Shoes"
            .tokenize()
            .embed(using: embeddings)
            .meanVector() else {
            XCTFail("meanVector returned nil")
            return
        }

        let results = embeddings.nearest(to: queryVector, k: 3)

        XCTAssertEqual(results.count, 3)
        // The three query words themselves should rank highest against their average.
        let topThree = Set(results.map { $0.word })
        XCTAssertTrue(topThree.contains("running"))
        XCTAssertTrue(topThree.contains("shoes"))
        XCTAssertTrue(topThree.contains("comfortable"))
    }
}
