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

final class ArrayStatisticsTests: XCTestCase {

    // Covers sum, product, cumulativeSum, cumulativeProduct
    func testAggregations() {
        let a = [1, 2, 3, 4, 5]
        XCTAssertEqual(a.sum(), 15)
        XCTAssertEqual(a.product(), 120)
        XCTAssertEqual(a.cumulativeSum(), [1, 3, 6, 10, 15])
        XCTAssertEqual(a.cumulativeProduct(), [1, 2, 6, 24, 120])
    }

    // Covers min, max, argMin, argMax
    func testMinMaxArg() {
        let a = [5, 3, 8, 1, 7]
        XCTAssertEqual(a.min(), 1)
        XCTAssertEqual(a.max(), 8)
        XCTAssertEqual(a.argMin(), 3)
        XCTAssertEqual(a.argMax(), 2)
    }

    // Covers mean, median, variance, standardDeviation
    func testCentralTendencyAndDispersion() throws {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0]
        XCTAssertEqual(a.mean(), 3.0)
        XCTAssertEqual(a.variance(), 2.5)

        let std = try XCTUnwrap(a.standardDeviation())
        XCTAssertEqual(std, sqrt(2.5), accuracy: 1e-10)

        let b = [1.0, 5.0, 3.0, 4.0, 2.0]
        XCTAssertEqual(b.median(), 3.0)
    }

    // MARK: - Outlier Detection Tests

    func testOutlierMaskBasic() {
        let data = [4.0, 7.0, 2.0, 9.0, 3.0, 35.0, 5.0]
        let mask = data.outlierMask(threshold: 2.0)

        XCTAssertEqual(mask.count, 7)
        XCTAssertEqual(mask[5], true)  // 35.0 - outlier
        XCTAssertEqual(mask.filter { $0 }.count, 1)
    }

    func testOutlierMaskPreCalculatedStats() {
        let data = [4.0, 7.0, 2.0, 9.0, 3.0, 35.0, 5.0]

        guard let mean = data.mean(), let std = data.standardDeviation() else {
            XCTFail("Unable to calculate statistics")
            return
        }

        let mask1 = data.outlierMask(threshold: 2.0)
        let mask2 = data.outlierMask(threshold: 2.0, mean: mean, standardDeviation: std)
        XCTAssertEqual(mask1, mask2)
    }

    func testOutlierMaskEdgeCases() {
        // Empty array
        XCTAssertEqual([Double]().outlierMask(threshold: 2.0), [])

        // Single element
        let single = [5.0].outlierMask(threshold: 2.0)
        XCTAssertEqual(single, [false])

        // No outliers
        let normal = [1.0, 2.0, 3.0, 4.0, 5.0]
        XCTAssertEqual(normal.outlierMask(threshold: 2.0).filter { $0 }.count, 0)
    }

    func testOutlierMaskWithMaskedBy() {
        let data = [4.0, 7.0, 2.0, 9.0, 3.0, 35.0, 5.0]
        let mask = data.outlierMask(threshold: 2.0)
        let outliers = data.masked(by: mask)

        XCTAssertEqual(outliers.count, 1)
        XCTAssertEqual(outliers.first, 35.0)
    }

    // MARK: - Vector Array Operations Tests

    // Covers meanVector basic and edge cases
    func testMeanVector() {
        let vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        let mean = vectors.meanVector()

        XCTAssertNotNil(mean)
        XCTAssertEqual(mean![0], 4.0, accuracy: 0.001)
        XCTAssertEqual(mean![1], 5.0, accuracy: 0.001)
        XCTAssertEqual(mean![2], 6.0, accuracy: 0.001)

        // Edge cases
        XCTAssertNil(([[Double]]()).meanVector())
        XCTAssertNil([[1.0, 2.0], [3.0, 4.0, 5.0]].meanVector())
    }

    // MARK: - Mode Tests

    func testModeSingleValueWins() {
        let rolls = [1, 3, 3, 5, 6, 3, 2]
        XCTAssertEqual(rolls.mode(), [3])
    }

    func testModeBimodalReturnsAllTies() {
        let ratings = [4, 5, 4, 3, 5, 4, 5]
        XCTAssertEqual(Set(ratings.mode()), Set([4, 5]))
    }

    func testModeAllUniqueReturnsAll() {
        let unique = [1, 2, 3, 4, 5]
        XCTAssertEqual(Set(unique.mode()), Set([1, 2, 3, 4, 5]))
    }

    func testModeEmptyReturnsEmpty() {
        let empty: [Int] = []
        XCTAssertEqual(empty.mode(), [])
    }

    func testModeOnDoubles() {
        let scores = [4.0, 5.0, 4.0, 3.0, 5.0, 4.0]
        XCTAssertEqual(scores.mode(), [4.0])
    }

    func testModeOnStrings() {
        let responses = ["yes", "no", "yes", "maybe", "yes", "no"]
        XCTAssertEqual(responses.mode(), ["yes"])
    }

    func testModeOnBools() {
        let workouts = [true, false, true, true, false, true, true]
        XCTAssertEqual(workouts.mode(), [true])
    }

    func testModeSingleElement() {
        let one = [42]
        XCTAssertEqual(one.mode(), [42])
    }

    // A non-finite entry corrupts the underlying percentile sort, so the
    // interval is rejected rather than reported as a silently wrong bound.
    func testPercentileCIRejectsNonFinite() {
        let withNaN = [1.0, 2.0, .nan, 4.0, 5.0]
        XCTAssertNil(withNaN.percentileCI(level: 0.95))

        let withInfinity = [1.0, 2.0, .infinity, 4.0, 5.0]
        XCTAssertNil(withInfinity.percentileCI(level: 0.95))

        // A clean distribution still produces an interval.
        let clean = [1.0, 2.0, 3.0, 4.0, 5.0]
        XCTAssertNotNil(clean.percentileCI(level: 0.95))
    }
}
