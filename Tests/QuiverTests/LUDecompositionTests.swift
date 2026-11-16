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

final class LUDecompositionTests: XCTestCase {

    // MARK: - Helpers

    /// Reconstructs `P · A` from the original matrix and the stored permutation.
    private func permute(_ a: [[Double]], by p: [Int]) -> [[Double]] {
        p.map { a[$0] }
    }

    /// Multiplies two square matrices.
    private func multiply(_ lhs: [[Double]], _ rhs: [[Double]]) -> [[Double]] {
        let n = lhs.count
        var result = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        for i in 0..<n {
            for j in 0..<n {
                var sum = 0.0
                for k in 0..<n { sum += lhs[i][k] * rhs[k][j] }
                result[i][j] = sum
            }
        }
        return result
    }

    private func assertMatrixEqual(_ lhs: [[Double]], _ rhs: [[Double]],
                                   accuracy: Double = 1e-9,
                                   file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(lhs.count, rhs.count, "row count", file: file, line: line)
        for i in 0..<lhs.count {
            XCTAssertEqual(lhs[i].count, rhs[i].count, "col count row \(i)", file: file, line: line)
            for j in 0..<lhs[i].count {
                XCTAssertEqual(lhs[i][j], rhs[i][j], accuracy: accuracy,
                               "mismatch at [\(i)][\(j)]", file: file, line: line)
            }
        }
    }

    // MARK: - P · A == L · U  (acceptance criterion 1)

    func testReconstructionNoPivotNeeded() throws {
        let a: [[Double]] = [[2, 1], [1, 3]]
        let lu = try a.luDecomposed()
        assertMatrixEqual(permute(a, by: lu.p), multiply(lu.l, lu.u))
    }

    func testReconstructionRequiresPivoting() throws {
        // A tiny leading pivot forces a row swap under partial pivoting.
        let a: [[Double]] = [[0.0001, 1], [1, 1]]
        let lu = try a.luDecomposed()
        assertMatrixEqual(permute(a, by: lu.p), multiply(lu.l, lu.u))
        // Confirm a swap actually happened.
        XCTAssertNotEqual(lu.p, [0, 1])
    }

    func testReconstruction3x3() throws {
        let a: [[Double]] = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]]
        let lu = try a.luDecomposed()
        assertMatrixEqual(permute(a, by: lu.p), multiply(lu.l, lu.u))
    }

    // MARK: - L is unit lower, U is upper triangular

    func testFactorShapes() throws {
        let a: [[Double]] = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]]
        let lu = try a.luDecomposed()
        let n = a.count
        for i in 0..<n {
            XCTAssertEqual(lu.l[i][i], 1.0, accuracy: 1e-12, "L diagonal must be 1")
            for j in 0..<n {
                if j > i { XCTAssertEqual(lu.l[i][j], 0.0, accuracy: 1e-12, "L upper must be 0") }
                if j < i { XCTAssertEqual(lu.u[i][j], 0.0, accuracy: 1e-12, "U lower must be 0") }
            }
        }
    }

    // MARK: - solve agrees with A.solve  (acceptance criterion 2)

    func testSolveMatchesMatrixSolve() throws {
        let a: [[Double]] = [[2, 1], [1, 3]]
        let b = [5.0, 10.0]
        let lu = try a.luDecomposed()
        let x = lu.solve(b)
        let reference = a.solve(b)!
        XCTAssertEqual(x[0], reference[0], accuracy: 1e-9)
        XCTAssertEqual(x[1], reference[1], accuracy: 1e-9)
        // Known solution.
        XCTAssertEqual(x[0], 1.0, accuracy: 1e-9)
        XCTAssertEqual(x[1], 3.0, accuracy: 1e-9)
    }

    func testSolve3x3KnownSolution() throws {
        let a: [[Double]] = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]]
        let lu = try a.luDecomposed()
        let x = lu.solve([5.0, -2.0, 9.0])
        XCTAssertEqual(x[0], 1.0, accuracy: 1e-9)
        XCTAssertEqual(x[1], 1.0, accuracy: 1e-9)
        XCTAssertEqual(x[2], 2.0, accuracy: 1e-9)
    }

    func testFactorOnceSolveMany() throws {
        // The headline use case: one factorization, several right-hand sides.
        let a: [[Double]] = [[2, 1], [1, 3]]
        let lu = try a.luDecomposed()
        for b in [[5.0, 10.0], [3.0, 4.0], [0.0, 0.0], [-1.0, 2.0]] {
            let x = lu.solve(b)
            // Verify A · x reconstructs b.
            let bx0 = 2 * x[0] + 1 * x[1]
            let bx1 = 1 * x[0] + 3 * x[1]
            XCTAssertEqual(bx0, b[0], accuracy: 1e-9)
            XCTAssertEqual(bx1, b[1], accuracy: 1e-9)
        }
    }

    // MARK: - determinant agrees, including sign  (acceptance criterion 3)

    func testDeterminantMatchesNoSwap() throws {
        let a: [[Double]] = [[2, 1], [1, 3]]   // det = 5
        let lu = try a.luDecomposed()
        XCTAssertEqual(lu.determinant, 5.0, accuracy: 1e-9)
        XCTAssertEqual(lu.determinant, a.determinant, accuracy: 1e-9)
    }

    func testDeterminantSignWithSwap() throws {
        // Swapping rows flips the determinant sign; partial pivoting here
        // forces a swap, and permutationSign must compensate.
        let a: [[Double]] = [[0.0001, 1], [1, 1]]   // det ≈ -0.9999
        let lu = try a.luDecomposed()
        XCTAssertEqual(lu.determinant, a.determinant, accuracy: 1e-9)
        XCTAssertLessThan(lu.determinant, 0)
    }

    func testDeterminant3x3() throws {
        let a: [[Double]] = [[2, 1, 1], [4, -6, 0], [-2, 7, 2]]
        let lu = try a.luDecomposed()
        XCTAssertEqual(lu.determinant, a.determinant, accuracy: 1e-9)
    }

    // MARK: - Error handling  (acceptance criterion 4)

    func testSingularThrows() {
        let a: [[Double]] = [[1, 2], [2, 4]]   // second row = 2 × first
        XCTAssertThrowsError(try a.luDecomposed()) { error in
            XCTAssertEqual(error as? MatrixError, .singular)
        }
    }

    func testNonSquareThrows() {
        let a: [[Double]] = [[1, 2, 3], [4, 5, 6]]
        XCTAssertThrowsError(try a.luDecomposed()) { error in
            XCTAssertEqual(error as? MatrixError, .notSquare)
        }
    }

    func testRaggedRowsThrowNotSquare() {
        let a: [[Double]] = [[1, 2], [3]]
        XCTAssertThrowsError(try a.luDecomposed()) { error in
            XCTAssertEqual(error as? MatrixError, .notSquare)
        }
    }

    func testEmptyThrowsNotSquare() {
        let a: [[Double]] = []
        XCTAssertThrowsError(try a.luDecomposed()) { error in
            XCTAssertEqual(error as? MatrixError, .notSquare)
        }
    }

    // MARK: - Scale-aware singular cutoff

    // Rank-deficient matrix with large entries throws even when rounding leaves a nonzero pivot.
    func testLargeMagnitudeRankDeficientThrows() {
        let sqft: [Double] = [1200, 1800, 2400, 1600, 2000, 2800, 1400, 2200, 1000, 3000]
        let design = sqft.map { [1.0, $0, $0 * 0.092903] }   // sqm column = sqft × 0.092903
        let gram = design.transposed().multiplyMatrix(design)
        XCTAssertThrowsError(try gram.luDecomposed()) { error in
            XCTAssertEqual(error as? MatrixError, .singular)
        }
        XCTAssertEqual(gram.determinant, 0.0)
    }

    // Well-conditioned matrix with tiny entries factors instead of tripping the cutoff.
    func testTinyMagnitudeIdentityIsNotSingular() throws {
        let tiny: [[Double]] = [[1e-20, 0, 0], [0, 1e-20, 0], [0, 0, 1e-20]]
        XCTAssertNoThrow(try tiny.luDecomposed())
        XCTAssertNotEqual(tiny.determinant, 0.0)
        XCTAssertEqual(tiny.conditionNumber, 1.0, accuracy: 1e-12)
    }

    // Closed-form 2×2 determinant classifies singular matrices the same way LU does.
    func testTwoByTwoDeterminantAgreesWithLU() {
        // Second column = 1.8 × first; the raw formula leaves a 1.86e-9 rounding residue.
        let nearlyDependent: [[Double]] = [[1234.5, 1234.5 * 1.8], [6789.1, 6789.1 * 1.8]]
        XCTAssertThrowsError(try nearlyDependent.luDecomposed())
        XCTAssertEqual(nearlyDependent.determinant, 0.0)
        XCTAssertEqual(nearlyDependent.logDeterminant.sign, 0)
    }

    // 2×2 singularity test stays correct where the raw products a·d and b·c underflow or overflow.
    func testTwoByTwoExtremeScalesAreNotSingular() throws {
        let tiny: [[Double]] = [[1e-200, 0], [0, 1e-200]]
        XCTAssertNoThrow(try tiny.luDecomposed())
        XCTAssertEqual(tiny.logDeterminant.sign, 1)
        XCTAssertEqual(tiny.logDeterminant.logAbsValue, 2 * Foundation.log(1e-200), accuracy: 1e-9)
        // Nonsingular, yet the determinant value itself underflows to zero.
        XCTAssertEqual(tiny.determinant, 0.0)

        let huge: [[Double]] = [[1e200, 0], [0, 1e200]]
        XCTAssertNoThrow(try huge.luDecomposed())
        XCTAssertEqual(huge.determinant, .infinity)
        XCTAssertEqual(huge.logDeterminant.sign, 1)
        XCTAssertEqual(huge.logDeterminant.logAbsValue, 2 * Foundation.log(1e200), accuracy: 1e-9)
    }

    // 2×2 log determinant carries the sign through a row swap and negative pivots.
    func testTwoByTwoLogDeterminantSign() {
        let swapped: [[Double]] = [[1, 2], [3, 4]]   // det = -2, elimination swaps rows
        XCTAssertEqual(swapped.logDeterminant.sign, -1)
        XCTAssertEqual(swapped.logDeterminant.logAbsValue, Foundation.log(2.0), accuracy: 1e-12)

        let negativePivots: [[Double]] = [[-3, 1], [1, -2]]   // det = 5, both pivots negative
        XCTAssertEqual(negativePivots.logDeterminant.sign, 1)
        XCTAssertEqual(negativePivots.logDeterminant.logAbsValue, Foundation.log(5.0), accuracy: 1e-12)
    }

}
