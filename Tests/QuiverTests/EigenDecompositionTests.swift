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

final class EigenDecompositionTests: XCTestCase {

    // Hilbert matrix with entries 1/(i + j + 1), matching the validation script
    private func hilbert(_ n: Int) -> [[Double]] {
        (0..<n).map { i in (0..<n).map { j in 1.0 / Double(i + j + 1) } }
    }

    // Maximum absolute residual of A·v - lambda·v for every eigenpair
    private func maxResidual(_ matrix: [[Double]], _ eigen: EigenDecomposition) -> Double {
        let n = matrix.count
        var worst = 0.0
        for k in 0..<n {
            let vector = eigen.eigenvectors[k]
            for i in 0..<n {
                var product = 0.0
                for j in 0..<n {
                    product += matrix[i][j] * vector[j]
                }
                worst = Swift.max(worst, abs(product - eigen.eigenvalues[k] * vector[i]))
            }
        }
        return worst
    }

    // Maximum deviation of the eigenvector rows from orthonormality
    private func maxOrthonormalityError(_ eigen: EigenDecomposition) -> Double {
        let n = eigen.eigenvectors.count
        var worst = 0.0
        for j in 0..<n {
            for k in 0..<n {
                var dot = 0.0
                for i in 0..<n {
                    dot += eigen.eigenvectors[j][i] * eigen.eigenvectors[k][i]
                }
                let expected = j == k ? 1.0 : 0.0
                worst = Swift.max(worst, abs(dot - expected))
            }
        }
        return worst
    }

    // Known analytic 2x2 with eigenvalues exactly 3 and 1
    func testKnownAnalyticEigenvalues() throws {
        let matrix: [[Double]] = [
            [2.0, 1.0],
            [1.0, 2.0]
        ]

        let eigen = try matrix.eigenDecomposed()

        XCTAssertEqual(eigen.eigenvalues[0], 3.0, accuracy: 1e-12)
        XCTAssertEqual(eigen.eigenvalues[1], 1.0, accuracy: 1e-12)
        XCTAssertTrue(eigen.converged)

        // Eigenvectors are (1,1)/sqrt(2) and (1,-1)/sqrt(2) after the sign rule
        let invSqrt2 = 1.0 / 2.0.squareRoot()
        XCTAssertEqual(eigen.eigenvectors[0][0], invSqrt2, accuracy: 1e-12)
        XCTAssertEqual(eigen.eigenvectors[0][1], invSqrt2, accuracy: 1e-12)
        XCTAssertEqual(eigen.eigenvectors[1][0], invSqrt2, accuracy: 1e-12)
        XCTAssertEqual(eigen.eigenvectors[1][1], -invSqrt2, accuracy: 1e-12)
    }

    // Identity input converges without dividing by zero; every eigenvalue is 1
    func testIdentityMatrix() throws {
        let identity: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]

        let eigen = try identity.eigenDecomposed()

        XCTAssertEqual(eigen.eigenvalues, [1.0, 1.0, 1.0])
        XCTAssertTrue(eigen.converged)

        // Each eigenvector is a standard basis vector
        for vector in eigen.eigenvectors {
            XCTAssertEqual(vector.filter { $0 == 1.0 }.count, 1)
            XCTAssertEqual(vector.filter { $0 == 0.0 }.count, 2)
        }
        XCTAssertEqual(maxOrthonormalityError(eigen), 0.0)
    }

    // Diagonal input converges in one sweep with exact eigenvalues and V = I
    func testDiagonalMatrixConvergesInOneSweep() throws {
        let diagonal: [[Double]] = [
            [5.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0]
        ]

        let eigen = try diagonal.eigenDecomposed()

        XCTAssertEqual(eigen.eigenvalues, [5.0, 3.0, 1.0])
        XCTAssertEqual(eigen.sweepsUsed, 1)
        XCTAssertTrue(eigen.converged)
        XCTAssertEqual(eigen.eigenvectors[0], [1.0, 0.0, 0.0])
        XCTAssertEqual(eigen.eigenvectors[1], [0.0, 1.0, 0.0])
        XCTAssertEqual(eigen.eigenvectors[2], [0.0, 0.0, 1.0])
    }

    // Repeated eigenvalue on a diagonal matrix: assert the invariant subspace,
    // never specific components of an eigenvector inside a repeated-eigenvalue subspace
    func testRepeatedEigenvalueDiagonal() throws {
        let matrix: [[Double]] = [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0]
        ]

        let eigen = try matrix.eigenDecomposed()

        XCTAssertEqual(eigen.eigenvalues[0], 3.0, accuracy: 1e-12)
        XCTAssertEqual(eigen.eigenvalues[1], 3.0, accuracy: 1e-12)
        XCTAssertEqual(eigen.eigenvalues[2], 1.0, accuracy: 1e-12)
        XCTAssertLessThanOrEqual(maxResidual(matrix, eigen), 1e-10)
    }

    // Cross-validated against NumPy (invariant-subspace projector, 1e-10 absolute)
    func testRepeatedEigenvalueNonDiagonal() throws {
        let matrix: [[Double]] = [
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ]

        let eigen = try matrix.eigenDecomposed()

        XCTAssertEqual(eigen.eigenvalues[0], 3.0, accuracy: 1e-12)
        XCTAssertEqual(eigen.eigenvalues[1], 3.0, accuracy: 1e-12)
        XCTAssertEqual(eigen.eigenvalues[2], 1.0, accuracy: 1e-12)

        // Projector onto the lambda = 3 subspace: P = v0·v0T + v1·v1T
        let expectedProjector: [[Double]] = [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0]
        ]
        for i in 0..<3 {
            for j in 0..<3 {
                var cell = 0.0
                for k in 0..<2 {
                    cell += eigen.eigenvectors[k][i] * eigen.eigenvectors[k][j]
                }
                XCTAssertEqual(cell, expectedProjector[i][j], accuracy: 1e-10,
                    "Projector mismatch at [\(i)][\(j)]")
            }
        }
        XCTAssertLessThanOrEqual(maxResidual(matrix, eigen), 1e-10)
    }

    // Cross-validated against NumPy (eigenvalues 1e-9 relative, eigenvectors 1e-8 absolute)
    func testGeneralMatrixMatchesReference() throws {
        let matrix: [[Double]] = [
            [4.0, 1.0, 0.5, 0.0],
            [1.0, 3.0, 0.25, 0.5],
            [0.5, 0.25, 5.0, 1.0],
            [0.0, 0.5, 1.0, 2.0]
        ]

        let expectedValues = [
            5.661111588438937,
            4.264439763580657,
            2.580732922597367,
            1.4937157253830389
        ]
        let expectedVectors: [[Double]] = [
            [0.41820516851205847, 0.28380994804279536, 0.8217510074352873, 0.2632140425601118],
            [0.7410599164187422, 0.4416693394351788, -0.49140726067661217, -0.11948765222669415],
            [-0.485126731188204, 0.7613649212024947, -0.14568104651823993, 0.4046634949366175],
            [0.20146139082686698, -0.3803992176084983, -0.24904059632668013, 0.8675762356301598]
        ]

        let eigen = try matrix.eigenDecomposed()

        for k in 0..<4 {
            XCTAssertEqual(eigen.eigenvalues[k] / expectedValues[k], 1.0, accuracy: 1e-9,
                "Eigenvalue \(k) outside relative tolerance")
            for i in 0..<4 {
                XCTAssertEqual(eigen.eigenvectors[k][i], expectedVectors[k][i], accuracy: 1e-8,
                    "Eigenvector \(k) component \(i) mismatch")
            }
        }
        XCTAssertTrue(eigen.converged)
    }

    // Cross-validated against NumPy (top eigenvalues, 1e-9 relative)
    func testIllConditionedHilbert() throws {
        let matrix = hilbert(6)

        // Top four eigenvalues at full tolerance; the smallest carry the
        // conditioning error and get a looser relative bound.
        let expectedValues = [
            1.6188998589243386,
            0.2423608705752095,
            0.016321521319875701,
            6.157483541825888e-04,
            1.2570757122630912e-05,
            1.0827994844022691e-07
        ]

        let eigen = try matrix.eigenDecomposed()

        for k in 0..<4 {
            XCTAssertEqual(eigen.eigenvalues[k] / expectedValues[k], 1.0, accuracy: 1e-9,
                "Top eigenvalue \(k) outside relative tolerance")
        }
        for k in 4..<6 {
            XCTAssertEqual(eigen.eigenvalues[k] / expectedValues[k], 1.0, accuracy: 1e-5,
                "Small eigenvalue \(k) outside loose relative tolerance")
        }

        // Reconstruction: sum of lambda_k * v_k v_kT recovers the matrix
        for i in 0..<6 {
            for j in 0..<6 {
                var cell = 0.0
                for k in 0..<6 {
                    cell += eigen.eigenvalues[k] * eigen.eigenvectors[k][i] * eigen.eigenvectors[k][j]
                }
                XCTAssertEqual(cell, matrix[i][j], accuracy: 1e-10,
                    "Reconstruction mismatch at [\(i)][\(j)]")
            }
        }

        // Orthonormality within the 4n·epsilon bound
        let bound = 4.0 * 6.0 * Double.ulpOfOne
        XCTAssertLessThanOrEqual(maxOrthonormalityError(eigen), bound)
        XCTAssertTrue(eigen.converged)
        XCTAssertLessThanOrEqual(eigen.sweepsUsed, 50)
    }

    // A covariance matrix from a constant column has a zero eigenvalue, no NaN
    func testConstantColumnCovariance() throws {
        let samples: [[Double]] = [
            [5.0, 10.0],
            [5.0, 20.0],
            [5.0, 30.0]
        ]
        let covariance = try XCTUnwrap(samples.covarianceMatrix())

        let eigen = try covariance.eigenDecomposed()

        XCTAssertEqual(eigen.eigenvalues[0], 100.0, accuracy: 1e-12)
        XCTAssertEqual(eigen.eigenvalues[1], 0.0, accuracy: 1e-12)
        XCTAssertFalse(eigen.eigenvalues.contains { $0.isNaN })
    }

    // MARK: - Edge-case shapes

    // The zero matrix returns zero eigenvalues and an identity basis
    func testZeroMatrix() throws {
        let zeros = [[Double]](repeating: [0.0, 0.0, 0.0], count: 3)

        let eigen = try zeros.eigenDecomposed()

        XCTAssertEqual(eigen.eigenvalues, [0.0, 0.0, 0.0])
        XCTAssertEqual(eigen.sweepsUsed, 0)
        XCTAssertTrue(eigen.converged)
        XCTAssertEqual(maxOrthonormalityError(eigen), 0.0)
    }

    // A 1x1 matrix returns its single entry and a unit eigenvector
    func testOneByOneMatrix() throws {
        let eigen = try [[7.5]].eigenDecomposed()

        XCTAssertEqual(eigen.eigenvalues, [7.5])
        XCTAssertEqual(eigen.eigenvectors, [[1.0]])
        XCTAssertTrue(eigen.converged)
    }

    // MARK: - Error channels

    // Empty, ragged, and non-square inputs throw notSquare
    func testShapeErrorsThrowNotSquare() {
        let empty: [[Double]] = []
        XCTAssertThrowsError(try empty.eigenDecomposed()) { error in
            XCTAssertEqual(error as? MatrixError, .notSquare)
        }

        let ragged: [[Double]] = [[1.0, 2.0], [3.0]]
        XCTAssertThrowsError(try ragged.eigenDecomposed()) { error in
            XCTAssertEqual(error as? MatrixError, .notSquare)
        }

        let rectangular: [[Double]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        XCTAssertThrowsError(try rectangular.eigenDecomposed()) { error in
            XCTAssertEqual(error as? MatrixError, .notSquare)
        }
    }

    // An asymmetric matrix throws notSymmetric; near-symmetric within tolerance passes
    func testAsymmetricThrowsNotSymmetric() throws {
        let asymmetric: [[Double]] = [[1.0, 2.0], [3.0, 4.0]]
        XCTAssertThrowsError(try asymmetric.eigenDecomposed()) { error in
            XCTAssertEqual(error as? MatrixError, .notSymmetric)
        }

        // A 1-ulp-scale mismatch passes the relative tolerance
        let nearSymmetric: [[Double]] = [
            [1.0, 2.0],
            [2.0 + 1e-13, 1.0]
        ]
        XCTAssertNoThrow(try nearSymmetric.eigenDecomposed())
    }

    // MARK: - Determinism

    // Identical input produces a bit-identical decomposition across reruns
    func testBitwiseDeterminism() throws {
        let matrix: [[Double]] = [
            [4.0, 1.0, 0.5, 0.0],
            [1.0, 3.0, 0.25, 0.5],
            [0.5, 0.25, 5.0, 1.0],
            [0.0, 0.5, 1.0, 2.0]
        ]

        let first = try matrix.eigenDecomposed()
        let second = try matrix.eigenDecomposed()

        XCTAssertEqual(first, second)
        XCTAssertEqual(first.sweepsUsed, second.sweepsUsed)
    }

    // MARK: - Equatable

    // Same input produces equal decompositions; different input differs
    func testEigenDecompositionEquatable() throws {
        let matrix: [[Double]] = [[2.0, 1.0], [1.0, 2.0]]

        let first = try matrix.eigenDecomposed()
        let second = try matrix.eigenDecomposed()
        XCTAssertEqual(first, second)

        let other = try [[5.0, 0.0], [0.0, 3.0]].eigenDecomposed()
        XCTAssertNotEqual(first, other)
    }

    // MARK: - Codable

    // Round-trip preserves equality and every stored property
    func testEigenDecompositionCodable() throws {
        let matrix: [[Double]] = [
            [4.0, 1.0, 0.5, 0.0],
            [1.0, 3.0, 0.25, 0.5],
            [0.5, 0.25, 5.0, 1.0],
            [0.0, 0.5, 1.0, 2.0]
        ]
        let eigen = try matrix.eigenDecomposed()

        let data = try JSONEncoder().encode(eigen)
        let decoded = try JSONDecoder().decode(EigenDecomposition.self, from: data)

        XCTAssertEqual(eigen, decoded)
        XCTAssertEqual(eigen.sweepsUsed, decoded.sweepsUsed)
        XCTAssertEqual(eigen.converged, decoded.converged)
    }

    // MARK: - Description

    // The printed form names the count and convergence in house format
    func testDescription() throws {
        let eigen = try [[2.0, 1.0], [1.0, 2.0]].eigenDecomposed()
        XCTAssertTrue(eigen.description.hasPrefix("EigenDecomposition: 2 eigenvalues, converged in"))

        let single = try [[7.5]].eigenDecomposed()
        XCTAssertTrue(single.description.hasPrefix("EigenDecomposition: 1 eigenvalue"))
    }
}
