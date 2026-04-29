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

final class FrequencyTests: XCTestCase {

    // MARK: - probability(of:) Tests

    // Canonical example: relative frequency over a small categorical sample
    func testProbabilityOfBasicCase() {
        let data = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0]
        XCTAssertEqual(data.probability(of: 1.0), 0.5, accuracy: 1e-12)
        XCTAssertEqual(data.probability(of: 2.0), 1.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(data.probability(of: 3.0), 1.0 / 6.0, accuracy: 1e-12)
    }

    // A value not in the array contributes zero matches
    func testProbabilityOfValueNotPresent() {
        let data = [1.0, 2.0, 3.0]
        XCTAssertEqual(data.probability(of: 99.0), 0.0)
    }

    // A constant array yields probability 1.0 for that value
    func testProbabilityOfAllSameValue() {
        let data = [7.0, 7.0, 7.0, 7.0]
        XCTAssertEqual(data.probability(of: 7.0), 1.0)
        XCTAssertEqual(data.probability(of: 8.0), 0.0)
    }

    // Empty array returns 0.0 — no division-by-zero
    func testProbabilityOfEmptyArray() {
        let data: [Double] = []
        XCTAssertEqual(data.probability(of: 1.0), 0.0)
    }

    // Single-element arrays give 1.0 on a match and 0.0 otherwise
    func testProbabilityOfSingleElement() {
        let match = [5.0]
        let nonMatch = [5.0]
        XCTAssertEqual(match.probability(of: 5.0), 1.0)
        XCTAssertEqual(nonMatch.probability(of: 4.0), 0.0)
    }

    // MARK: - frequencyDistribution() Tests

    // Basic case — exact ratios for a six-element categorical sample
    func testFrequencyDistributionBasicCase() {
        let data = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0]
        let dist = data.frequencyDistribution()

        XCTAssertEqual(dist.count, 3)
        XCTAssertEqual(dist[1.0] ?? 0.0, 0.5, accuracy: 1e-12)
        XCTAssertEqual(dist[2.0] ?? 0.0, 1.0 / 3.0, accuracy: 1e-12)
        XCTAssertEqual(dist[3.0] ?? 0.0, 1.0 / 6.0, accuracy: 1e-12)
    }

    // Frequencies must sum to 1.0 within floating-point tolerance
    func testFrequencyDistributionSumsToOne() {
        let data = [1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 4.0, 5.0, 4.0]
        let dist = data.frequencyDistribution()
        let total = dist.values.reduce(0.0, +)
        XCTAssertEqual(total, 1.0, accuracy: 1e-12)
    }

    // Empty array returns an empty dictionary, not a crash
    func testFrequencyDistributionEmptyArray() {
        let data: [Double] = []
        XCTAssertEqual(data.frequencyDistribution(), [:])
    }

    // A constant array maps that value to 1.0
    func testFrequencyDistributionAllSameValue() {
        let data = [7.0, 7.0, 7.0]
        let dist = data.frequencyDistribution()
        XCTAssertEqual(dist.count, 1)
        XCTAssertEqual(dist[7.0] ?? 0.0, 1.0, accuracy: 1e-12)
    }

    // Single-element array maps the lone value to 1.0
    func testFrequencyDistributionSingleElement() {
        let data = [42.0]
        let dist = data.frequencyDistribution()
        XCTAssertEqual(dist.count, 1)
        XCTAssertEqual(dist[42.0] ?? 0.0, 1.0, accuracy: 1e-12)
    }

    // MARK: - distinct() Tests

    // [Double] — verify ascending sort
    func testDistinctDoubleAscendingSort() {
        let data = [3.0, 1.0, 2.0, 1.0]
        XCTAssertEqual(data.distinct(), [1.0, 2.0, 3.0])
    }

    // [Int] — generic constraint compiles and works
    func testDistinctInt() {
        let data = [5, 3, 5, 1, 3, 2]
        XCTAssertEqual(data.distinct(), [1, 2, 3, 5])
    }

    // [String] — generic constraint compiles and works
    func testDistinctString() {
        let data = ["beta", "alpha", "alpha", "gamma"]
        XCTAssertEqual(data.distinct(), ["alpha", "beta", "gamma"])
    }

    // Already-unique input returns the sorted version
    func testDistinctAlreadyUnique() {
        let data = [4.0, 2.0, 1.0, 3.0]
        XCTAssertEqual(data.distinct(), [1.0, 2.0, 3.0, 4.0])
    }

    // All-duplicate input collapses to a single-element array
    func testDistinctAllDuplicates() {
        let data = [9.0, 9.0, 9.0, 9.0]
        XCTAssertEqual(data.distinct(), [9.0])
    }

    // Empty array returns []
    func testDistinctEmpty() {
        let data: [Double] = []
        XCTAssertEqual(data.distinct(), [])
    }

    // Determinism — repeated calls produce identical output. This is the key
    // win over Array(Set(x)), whose ordering is not stable across runs.
    func testDistinctIsDeterministic() {
        let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]
        let firstCall = data.distinct()
        let secondCall = data.distinct()
        XCTAssertEqual(firstCall, secondCall)
        XCTAssertEqual(firstCall, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0])
    }

    // MARK: - distinctCounts() Tests

    // Basic case — verify both values and counts in ascending value order
    func testDistinctCountsBasicCase() {
        let data = [3.0, 1.0, 2.0, 1.0]
        let table = data.distinctCounts()

        XCTAssertEqual(table.count, 3)
        XCTAssertEqual(table[0].value, 1.0)
        XCTAssertEqual(table[0].count, 2)
        XCTAssertEqual(table[1].value, 2.0)
        XCTAssertEqual(table[1].count, 1)
        XCTAssertEqual(table[2].value, 3.0)
        XCTAssertEqual(table[2].count, 1)
    }

    // The tuple shape is (value:, count:) and accessible by labels
    func testDistinctCountsTupleLabels() {
        let data = [10.0, 20.0, 10.0]
        let table = data.distinctCounts()
        guard let first = table.first else {
            XCTFail("Expected at least one entry")
            return
        }
        // Access by label, not by index — confirms the labeled tuple shape
        XCTAssertEqual(first.value, 10.0)
        XCTAssertEqual(first.count, 2)
    }

    // Empty array returns []
    func testDistinctCountsEmpty() {
        let data: [Double] = []
        let table = data.distinctCounts()
        XCTAssertEqual(table.count, 0)
    }

    // All-same input produces a single tuple
    func testDistinctCountsAllSame() {
        let data = [5.0, 5.0, 5.0, 5.0]
        let table = data.distinctCounts()
        XCTAssertEqual(table.count, 1)
        XCTAssertEqual(table[0].value, 5.0)
        XCTAssertEqual(table[0].count, 4)
    }

    // Generic over Hashable & Comparable — exercise the [String] case for parity
    // with distinct()
    func testDistinctCountsStringElement() {
        let words = ["beta", "alpha", "alpha", "gamma", "alpha"]
        let table = words.distinctCounts()
        XCTAssertEqual(table.count, 3)
        XCTAssertEqual(table[0].value, "alpha")
        XCTAssertEqual(table[0].count, 3)
        XCTAssertEqual(table[1].value, "beta")
        XCTAssertEqual(table[1].count, 1)
        XCTAssertEqual(table[2].value, "gamma")
        XCTAssertEqual(table[2].count, 1)
    }
}
