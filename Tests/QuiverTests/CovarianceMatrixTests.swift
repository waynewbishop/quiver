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

final class CovarianceMatrixTests: XCTestCase {

    // Perfectly correlated columns produce the known analytic covariance
    func testKnownAnalyticCovariance() {
        let samples: [[Double]] = [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0]
        ]

        let covariance = samples.covarianceMatrix()

        XCTAssertNotNil(covariance)
        XCTAssertEqual(covariance?[0][0] ?? .nan, 1.0, accuracy: 1e-12)
        XCTAssertEqual(covariance?[0][1] ?? .nan, 2.0, accuracy: 1e-12)
        XCTAssertEqual(covariance?[1][0] ?? .nan, 2.0, accuracy: 1e-12)
        XCTAssertEqual(covariance?[1][1] ?? .nan, 4.0, accuracy: 1e-12)
    }

    // Cross-validated against NumPy (cells, 1e-12 absolute)
    func testSampleCovarianceMatchesReference() throws {
        let samples: [[Double]] = [
            [2.5, 24.0, 0.31],
            [0.5, 11.0, 0.20],
            [2.2, 22.0, 0.28],
            [1.9, 20.0, 0.25],
            [3.1, 30.0, 0.36]
        ]

        let expected: [[Double]] = [
            [0.938, 6.655, 0.0565],
            [6.655, 47.8, 0.41],
            [0.0565, 0.41, 0.00365]
        ]

        let covariance = try XCTUnwrap(samples.covarianceMatrix())

        for j in 0..<3 {
            for k in 0..<3 {
                XCTAssertEqual(covariance[j][k], expected[j][k], accuracy: 1e-12,
                    "Mismatch at cell [\(j)][\(k)]")
            }
        }
    }

    // Cross-validated against NumPy (cells, 1e-12 absolute)
    func testPopulationCovarianceMatchesReference() throws {
        let samples: [[Double]] = [
            [2.5, 24.0, 0.31],
            [0.5, 11.0, 0.20],
            [2.2, 22.0, 0.28],
            [1.9, 20.0, 0.25],
            [3.1, 30.0, 0.36]
        ]

        let expected: [[Double]] = [
            [0.7504, 5.324, 0.0452],
            [5.324, 38.24, 0.328],
            [0.0452, 0.328, 0.00292]
        ]

        let covariance = try XCTUnwrap(samples.covarianceMatrix(ddof: 0))

        for j in 0..<3 {
            for k in 0..<3 {
                XCTAssertEqual(covariance[j][k], expected[j][k], accuracy: 1e-12,
                    "Mismatch at cell [\(j)][\(k)]")
            }
        }
    }

    // The diagonal matches variance(ddof:) computed per column
    func testDiagonalMatchesColumnVariance() throws {
        let samples: [[Double]] = [
            [2.5, 24.0, 0.31],
            [0.5, 11.0, 0.20],
            [2.2, 22.0, 0.28],
            [1.9, 20.0, 0.25],
            [3.1, 30.0, 0.36]
        ]

        let covariance = try XCTUnwrap(samples.covarianceMatrix())

        for column in 0..<3 {
            let values = samples.map { $0[column] }
            let variance = try XCTUnwrap(values.variance(ddof: 1))
            XCTAssertEqual(covariance[column][column], variance, accuracy: 1e-12)
        }
    }

    // Mirrored cells are bitwise identical, not merely close
    func testExactSymmetry() throws {
        let samples: [[Double]] = [
            [0.31, 2.5, 24.0, 7.7],
            [0.20, 0.5, 11.0, 3.2],
            [0.28, 2.2, 22.0, 6.1],
            [0.25, 1.9, 20.0, 5.9],
            [0.36, 3.1, 30.0, 8.8]
        ]

        let covariance = try XCTUnwrap(samples.covarianceMatrix())

        for j in 0..<4 {
            for k in 0..<4 {
                XCTAssertEqual(covariance[j][k], covariance[k][j],
                    "Cells [\(j)][\(k)] and [\(k)][\(j)] are not bitwise equal")
            }
        }
    }

    // A constant column produces an exactly zero covariance row and column
    func testConstantColumnGivesExactZeros() throws {
        let samples: [[Double]] = [
            [5.0, 10.0],
            [5.0, 20.0],
            [5.0, 30.0]
        ]

        let covariance = try XCTUnwrap(samples.covarianceMatrix())

        XCTAssertEqual(covariance[0][0], 0.0)
        XCTAssertEqual(covariance[0][1], 0.0)
        XCTAssertEqual(covariance[1][0], 0.0)
        XCTAssertFalse(covariance.contains { row in row.contains { $0.isNaN } })

        // Column 1: [10, 20, 30] has sample variance 100
        XCTAssertEqual(covariance[1][1], 100.0, accuracy: 1e-12)
    }

    // MARK: - Edge cases

    // Empty input returns nil
    func testEmptyInputReturnsNil() {
        let samples: [[Double]] = []
        XCTAssertNil(samples.covarianceMatrix())
    }

    // Ragged rows return nil
    func testRaggedRowsReturnNil() {
        let samples: [[Double]] = [
            [1.0, 2.0],
            [3.0]
        ]
        XCTAssertNil(samples.covarianceMatrix())
    }

    // Fewer than ddof + 1 rows returns nil
    func testInsufficientRowsReturnNil() {
        let single: [[Double]] = [[1.0, 2.0]]
        XCTAssertNil(single.covarianceMatrix())
        XCTAssertNil(single.covarianceMatrix(ddof: 1))

        // One row is enough for the population covariance, which is all zeros
        let population = single.covarianceMatrix(ddof: 0)
        XCTAssertEqual(population?[0][0], 0.0)
        XCTAssertEqual(population?[0][1], 0.0)
    }

    // Negative ddof returns nil
    func testNegativeDdofReturnsNil() {
        let samples: [[Double]] = [
            [1.0, 2.0],
            [2.0, 4.0]
        ]
        XCTAssertNil(samples.covarianceMatrix(ddof: -1))
    }
}
