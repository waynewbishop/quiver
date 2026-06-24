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

final class EmbeddingIndexTests: XCTestCase {

    // MARK: - Test conformer

    /// A deterministic embedder: text maps to a fixed vector through a table, so
    /// every test is reproducible without a real embedding model. Unknown text
    /// returns nil — exactly the "unembeddable" signal the index must handle.
    private struct StubEmbedder: Embedder {
        let table: [String: [Double]]
        func embed(_ text: String) -> [Double]? { table[text] }
    }

    private var embedder: StubEmbedder {
        StubEmbedder(table: [
            // four orthogonal-ish directions so cosine ranking is unambiguous
            "cat":   [1.0, 0.0, 0.0],
            "dog":   [0.9, 0.1, 0.0],   // close to cat
            "car":   [0.0, 1.0, 0.0],
            "truck": [0.0, 0.9, 0.1],   // close to car
            "sky":   [0.0, 0.0, 1.0],
            "":      [0.0, 0.0, 0.0],   // empty embeds to zero
            "unknown-query": [0.3, 0.3, 0.3],
        ])
    }

    private func makeIndex() -> EmbeddingIndex<String> {
        var index = EmbeddingIndex<String>(embedder: embedder)
        index.add("cat", label: "cat")
        index.add("dog", label: "dog")
        index.add("car", label: "car")
        index.add("truck", label: "truck")
        index.add("sky", label: "sky")
        return index
    }

    // MARK: - Ingest

    func testAddStoresEntries() {
        let index = makeIndex()
        XCTAssertEqual(index.count, 5)
        XCTAssertFalse(index.isEmpty)
    }

    func testUnembeddableTextIsSkipped() {
        var index = EmbeddingIndex<String>(embedder: embedder)
        index.add("cat", label: "cat")
        index.add("not-in-table", label: "ignored")   // embed returns nil
        XCTAssertEqual(index.count, 1)
    }

    func testAddVectorStoresWithoutEmbedding() {
        var index = EmbeddingIndex<String>(embedder: embedder)
        index.add(vector: [0.5, 0.5, 0.0], label: "manual")
        XCTAssertEqual(index.count, 1)
        XCTAssertEqual(index.vector(for: "manual"), [0.5, 0.5, 0.0])
    }

    // MARK: - Retrieve

    func testRetrieveRanksClosestFirst() {
        let index = makeIndex()
        let result = index.retrieve("cat", k: 2)

        XCTAssertEqual(result.hits.count, 2)
        XCTAssertEqual(result.hits[0].label, "cat")   // identical → score 1.0
        XCTAssertEqual(result.hits[0].rank, 1)
        XCTAssertEqual(result.hits[0].score, 1.0, accuracy: 1e-9)
        XCTAssertEqual(result.hits[1].label, "dog")   // nearest neighbor
        XCTAssertGreaterThan(result.hits[0].score, result.hits[1].score)
    }

    func testRetrieveReturnsFullScoreField() {
        let index = makeIndex()
        let result = index.retrieve("cat", k: 2)
        // hits are top-2, but scores is the WHOLE field — all 5 entries.
        XCTAssertEqual(result.hits.count, 2)
        XCTAssertEqual(result.scores.count, 5)
    }

    func testRetrieveEmptyIndexYieldsEmptyResult() {
        let index = EmbeddingIndex<String>(embedder: embedder)
        let result = index.retrieve("cat", k: 3)
        XCTAssertTrue(result.hits.isEmpty)
        XCTAssertTrue(result.scores.isEmpty)
    }

    func testRetrieveUnembeddableQueryYieldsEmptyResult() {
        let index = makeIndex()
        let result = index.retrieve("not-in-table", k: 3)   // query embeds to nil
        XCTAssertTrue(result.hits.isEmpty)
    }

    // MARK: - The math is exposed

    func testResultCarriesDistributionStats() {
        let index = makeIndex()
        let result = index.retrieve("cat", k: 3)
        // mean/spread computed over the full field, not just the hits.
        let expectedMean = result.scores.reduce(0, +) / Double(result.scores.count)
        XCTAssertEqual(result.mean, expectedMean, accuracy: 1e-9)
        XCTAssertGreaterThan(result.standardDeviation, 0)   // field has spread
        XCTAssertEqual(result.topScore, result.scores.max()!, accuracy: 1e-9)
    }

    func testTopZScoreMatchesManualComputation() {
        let index = makeIndex()
        let result = index.retrieve("cat", k: 3)
        let manualZ = result.scores.zScore(of: result.topScore) ?? 0
        XCTAssertEqual(result.topZScore, manualZ, accuracy: 1e-9)
    }

    func testVectorsExposesCorpusMatrix() {
        let index = makeIndex()
        XCTAssertEqual(index.vectors.count, 5)
        // pairwise cosine reachable directly — the transparency guarantee
        let simCatDog = index.vectors[0].cosineOfAngle(with: index.vectors[1])
        XCTAssertGreaterThan(simCatDog, 0.9)   // cat & dog are close
    }

    func testScoresForExposesFullFieldWithoutRanking() {
        let index = makeIndex()
        let scores = index.scores(for: "cat")
        XCTAssertEqual(scores.count, 5)
        XCTAssertEqual(scores.max()!, 1.0, accuracy: 1e-9)   // cat vs cat
    }

    func testVectorForLabel() {
        let index = makeIndex()
        XCTAssertEqual(index.vector(for: "sky"), [0.0, 0.0, 1.0])
        XCTAssertNil(index.vector(for: "missing"))
    }

    // MARK: - The gate (caller's thresholds)

    func testIsAboveGatePassesViaFloor() {
        let index = makeIndex()
        let result = index.retrieve("cat", k: 3)   // top score 1.0
        XCTAssertTrue(result.isAboveGate(floor: 0.30, outlierZ: 3.0))
    }

    func testIsAboveGateFailsWhenNeitherBarMet() {
        let index = makeIndex()
        let result = index.retrieve("cat", k: 3)
        // floor above the top score AND z bar above the top z → fails both
        XCTAssertFalse(result.isAboveGate(floor: 1.1, outlierZ: 99.0))
    }

    func testIsAboveGateEmptyResultIsFalse() {
        let index = EmbeddingIndex<String>(embedder: embedder)
        let result = index.retrieve("cat", k: 3)
        XCTAssertFalse(result.isAboveGate(floor: 0.0, outlierZ: 0.0))
    }

    // MARK: - Removal (closure-free at the API)

    func testRemoveLabel() {
        var index = makeIndex()
        index.remove(label: "dog")
        XCTAssertEqual(index.count, 4)
        XCTAssertNil(index.vector(for: "dog"))
    }

    func testRemoveLabels() {
        var index = makeIndex()
        index.remove(labels: ["dog", "truck"])
        XCTAssertEqual(index.count, 3)
    }

    func testRemoveAll() {
        var index = makeIndex()
        index.removeAll()
        XCTAssertTrue(index.isEmpty)
    }

    // MARK: - Sequence

    func testSequenceIteration() {
        let index = makeIndex()
        var labels: [String] = []
        for entry in index {
            labels.append(entry.label)
        }
        XCTAssertEqual(labels, ["cat", "dog", "car", "truck", "sky"])
    }

    // MARK: - Persistence

    func testSnapshotEncodeDecodeRoundTrip() throws {
        let index = makeIndex()

        // Encode/decode the snapshot with stdlib JSON, the same pattern models use.
        let data = try JSONEncoder().encode(index.snapshot)
        let snapshot = try JSONDecoder().decode(EmbeddingIndex<String>.Snapshot.self, from: data)
        let restored = EmbeddingIndex(snapshot, embedder: embedder)

        XCTAssertEqual(restored.count, index.count)
        XCTAssertEqual(restored.vectors, index.vectors)
        // a restored index retrieves identically
        let original = index.retrieve("cat", k: 2).hits.map(\.label)
        let again = restored.retrieve("cat", k: 2).hits.map(\.label)
        XCTAssertEqual(original, again)
    }

    func testSnapshotIsEquatable() throws {
        let index = makeIndex()
        XCTAssertEqual(index.snapshot, index.snapshot)
        var other = makeIndex()
        other.add("dog", label: "extra")
        XCTAssertNotEqual(index.snapshot, other.snapshot)
    }

    // MARK: - RetrievedHit / RetrievalResult value semantics

    func testRetrievedHitEquatableAndCodable() throws {
        let hit = RetrievedHit(rank: 1, label: "cat", score: 0.91)
        let data = try JSONEncoder().encode(hit)
        let decoded = try JSONDecoder().decode(RetrievedHit<String>.self, from: data)
        XCTAssertEqual(hit, decoded)
    }
}
