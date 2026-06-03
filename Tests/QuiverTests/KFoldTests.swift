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
@testable import Quiver

final class KFoldTests: XCTestCase {

    // A stand-in data array; only its count matters for fold generation.
    private func data(_ n: Int) -> [Int] { Array(0..<n) }

    // MARK: - Fold count

    func testReturnsKFolds() {
        let folds = data(10).kFoldIndices(k: 5, seed: 42)
        XCTAssertEqual(folds.count, 5)
    }

    // MARK: - Complete, disjoint validation coverage (the core invariant)

    // Every sample index must appear in exactly one validation fold — complete
    // coverage with no overlap. This is the property that proves the splitter is
    // sound.
    func testEveryIndexValidatedExactlyOnce() {
        let n = 10
        let folds = data(n).kFoldIndices(k: 5, seed: 7)

        var seen = [Int: Int]()   // index -> how many validation folds contain it
        for fold in folds {
            for idx in fold.validation {
                seen[idx, default: 0] += 1
            }
        }

        // Every index 0..<n appears exactly once.
        XCTAssertEqual(seen.count, n, "every index must be validated")
        for idx in 0..<n {
            XCTAssertEqual(seen[idx], 1, "index \(idx) must appear in exactly one validation fold")
        }
    }

    // Within each fold, train and validation must partition the full index set:
    // disjoint, and together covering everything.
    func testTrainAndValidationPartitionEachFold() {
        let n = 12
        let folds = data(n).kFoldIndices(k: 4, seed: 99)
        let all = Set(0..<n)

        for (i, fold) in folds.enumerated() {
            let train = Set(fold.train)
            let validation = Set(fold.validation)

            XCTAssertTrue(train.isDisjoint(with: validation),
                "fold \(i): train and validation must not overlap")
            XCTAssertEqual(train.union(validation), all,
                "fold \(i): train ∪ validation must cover every index")
            XCTAssertEqual(train.count + validation.count, n,
                "fold \(i): no index duplicated within a fold")
        }
    }

    // MARK: - Balanced fold sizes

    // When count is divisible by k, every validation fold is the same size.
    func testEvenFoldSizes() {
        let folds = data(10).kFoldIndices(k: 5, seed: 1)
        for fold in folds {
            XCTAssertEqual(fold.validation.count, 2)
            XCTAssertEqual(fold.train.count, 8)
        }
    }

    // When count is not divisible by k, validation sizes differ by at most one.
    func testUnevenFoldSizesDifferByAtMostOne() {
        // 11 samples, 3 folds -> sizes 4, 4, 3.
        let folds = data(11).kFoldIndices(k: 3, seed: 1)
        let sizes = folds.map { $0.validation.count }.sorted()
        XCTAssertEqual(sizes, [3, 4, 4])
        XCTAssertLessThanOrEqual(sizes.max()! - sizes.min()!, 1)
    }

    // MARK: - Reproducibility

    func testSameSeedProducesSameFolds() {
        let a = data(20).kFoldIndices(k: 5, seed: 12345)
        let b = data(20).kFoldIndices(k: 5, seed: 12345)

        XCTAssertEqual(a.count, b.count)
        for (fa, fb) in zip(a, b) {
            XCTAssertEqual(fa.train, fb.train)
            XCTAssertEqual(fa.validation, fb.validation)
        }
    }

    func testDifferentSeedProducesDifferentFolds() {
        let a = data(20).kFoldIndices(k: 5, seed: 1)
        let b = data(20).kFoldIndices(k: 5, seed: 2)

        // Validation membership should differ for at least one fold.
        let aVal = a.map { Set($0.validation) }
        let bVal = b.map { Set($0.validation) }
        XCTAssertNotEqual(aVal, bVal)
    }

    // MARK: - Shuffle, not sequential

    // The folds must reflect a shuffle, not naive contiguous slices of 0..<n —
    // otherwise fold 0's validation would always be [0, 1] and cross-validation
    // would be biased by input order.
    func testFoldsAreShuffledNotSequential() {
        let folds = data(20).kFoldIndices(k: 4, seed: 555)
        let firstValidation = folds[0].validation
        XCTAssertNotEqual(firstValidation.sorted(), Array(0..<firstValidation.count),
            "validation should be a shuffled subset, not the leading indices")
    }

    // MARK: - Edge cases

    // k == count is leave-one-out: each validation fold holds exactly one index.
    func testLeaveOneOut() {
        let n = 6
        let folds = data(n).kFoldIndices(k: n, seed: 3)
        XCTAssertEqual(folds.count, n)
        for fold in folds {
            XCTAssertEqual(fold.validation.count, 1)
            XCTAssertEqual(fold.train.count, n - 1)
        }
        // Still complete coverage.
        let validated = Set(folds.flatMap { $0.validation })
        XCTAssertEqual(validated, Set(0..<n))
    }

    func testTwoFolds() {
        let folds = data(8).kFoldIndices(k: 2, seed: 1)
        XCTAssertEqual(folds.count, 2)
        XCTAssertEqual(folds[0].validation.count, 4)
        XCTAssertEqual(folds[1].validation.count, 4)
    }

    // MARK: - Application to parallel arrays

    // The indices must address parallel arrays consistently — the canonical use.
    func testIndicesApplyToParallelArrays() {
        let features = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        let targets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

        let folds = features.kFoldIndices(k: 3, seed: 8)
        for fold in folds {
            let trainF = fold.train.map { features[$0] }
            let trainT = fold.train.map { targets[$0] }
            // Same index count on both sides, and the pairing is preserved:
            // targets[i] == features[i][0] * 10 for our fixture.
            XCTAssertEqual(trainF.count, trainT.count)
            for (f, t) in zip(trainF, trainT) {
                XCTAssertEqual(f[0] * 10.0, t, accuracy: 1e-9)
            }
        }
    }
}
