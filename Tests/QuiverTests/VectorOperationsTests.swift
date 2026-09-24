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

final class VectorOperationsTests: XCTestCase {

    // MARK: - Basic Vector Operations

    func testDotProduct() {
        let a = [1.0, 2.0, 3.0]
        let b = [4.0, 5.0, 6.0]
        let result = a.dot(b)
        XCTAssertEqual(result, 32.0)
    }

    func testMagnitude() {
        let v = [3.0, 4.0]
        XCTAssertEqual(v.magnitude, 5.0)
    }

    func testNormalized() {
        let v = [3.0, 0.0]
        let result = v.normalized
        XCTAssertEqual(result, [1.0, 0.0])
    }

    // MARK: - Cosine and Angle Tests

    func testCosineOfAngle() {
        // 45 degrees
        let v1 = [1.0, 0.0]
        let v2 = [1.0, 1.0]
        let cosine = v1.cosineOfAngle(with: v2)
        XCTAssertEqual(cosine, 1.0/sqrt(2.0), accuracy: 1e-10)

        // Perpendicular vectors
        let v3 = [0.0, 1.0]
        XCTAssertEqual(v1.cosineOfAngle(with: v3), 0.0, accuracy: 1e-10)

        // Parallel vectors
        let v4 = [3.0, 4.0]
        let v5 = [6.0, 8.0]
        XCTAssertEqual(v4.cosineOfAngle(with: v5), 1.0, accuracy: 1e-10)
    }

    func testCorrelation() {
        // Standard worked example
        let x = [1.0, 2.0, 3.0, 4.0, 5.0]
        let y = [2.0, 4.0, 5.0, 4.0, 5.0]
        XCTAssertEqual(x.correlation(with: y)!, 0.7745966692, accuracy: 1e-10)

        // Perfect positive linear relationship
        let a = [1.0, 2.0, 3.0]
        let b = [2.0, 4.0, 6.0]
        XCTAssertEqual(a.correlation(with: b)!, 1.0, accuracy: 1e-10)

        // Perfect negative linear relationship
        let c = [6.0, 4.0, 2.0]
        XCTAssertEqual(a.correlation(with: c)!, -1.0, accuracy: 1e-10)

        // Symmetric in its arguments
        XCTAssertEqual(x.correlation(with: y)!, y.correlation(with: x)!, accuracy: 1e-10)

        // Shift invariance — adding a constant to either vector leaves r unchanged
        let yShifted = y.map { $0 + 100.0 }
        XCTAssertEqual(x.correlation(with: y)!, x.correlation(with: yShifted)!, accuracy: 1e-10)

        // Constant column produces nil — variance is zero and the ratio is undefined
        let constant = [3.0, 3.0, 3.0]
        let firstThree = Array(x.prefix(3))
        XCTAssertNil(firstThree.correlation(with: constant))

        // Agrees with the matrix-level convenience for the same pair
        let matrix = [x, y].correlationMatrix()
        XCTAssertEqual(x.correlation(with: y)!, matrix[0][1], accuracy: 1e-10)

        // Empty and mismatched lengths return nil
        XCTAssertNil([Double]().correlation(with: []))
        XCTAssertNil([1.0, 2.0].correlation(with: [1.0]))
    }

    func testCosineSimilarities() {
        let vectors = [
            [1.0, 0.0],      // Along x-axis
            [0.0, 1.0]       // Along y-axis (perpendicular)
        ]
        let target = [1.0, 0.0]

        let similarities = vectors.cosineSimilarities(to: target)

        XCTAssertEqual(similarities[0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(similarities[1], 0.0, accuracy: 1e-10)

        // Empty array
        let empty: [[Double]] = []
        XCTAssertTrue(empty.cosineSimilarities(to: target).isEmpty)
    }

    // MARK: - Projection Tests

    func testScalarProjection() {
        let v = [3.0, 4.0]
        let axis = [1.0, 0.0]
        let result = v.scalarProjection(onto: axis)
        XCTAssertEqual(result, 3.0)
    }

    func testVectorProjection() {
        let v = [3.0, 4.0]
        let axis = [1.0, 0.0]
        let result = v.vectorProjection(onto: axis)
        XCTAssertEqual(result, [3.0, 0.0])
    }

    func testOrthogonalComponent() {
        let v = [3.0, 4.0]
        let axis = [1.0, 0.0]
        let result = v.orthogonalComponent(to: axis)
        XCTAssertEqual(result, [0.0, 4.0])
    }

    // A zero reference vector has no direction, so projection contributes
    // nothing. These are the stationary-runner and body-at-rest cases: the
    // convention keeps them out of a trap in a per-frame sampling loop.
    func testScalarProjectionOntoZeroVectorReturnsZero() {
        let v = [3.0, 4.0]
        XCTAssertEqual(v.scalarProjection(onto: [0.0, 0.0]), 0.0)
    }

    func testVectorProjectionOntoZeroVectorReturnsZeroVector() {
        let v = [3.0, 4.0]
        XCTAssertEqual(v.vectorProjection(onto: [0.0, 0.0]), [0.0, 0.0])
    }

    func testOrthogonalComponentOfZeroVectorReturnsSelf() {
        let v = [3.0, 4.0]
        XCTAssertEqual(v.orthogonalComponent(to: [0.0, 0.0]), [3.0, 4.0])
    }

    // The defining identity: the parallel and perpendicular parts sum back
    // to the original. It must survive the zero case, which is what makes
    // the convention correct rather than merely convenient.
    func testProjectionIdentityHoldsForZeroReference() {
        let v = [3.0, 4.0]
        let zero = [0.0, 0.0]
        let parallel = v.vectorProjection(onto: zero)
        let perpendicular = v.orthogonalComponent(to: zero)
        XCTAssertEqual(parallel.add(perpendicular), v)
    }

    // magnitude() sums squares before taking a square root, so a reference
    // this small underflows to zero magnitude and takes the convention path
    // rather than dividing. Both methods share that one guard, so they agree
    // on where the boundary falls instead of disagreeing near it.
    func testProjectionsAgreeOnUnderflowingReference() {
        let v = [3.0, 4.0]
        let tiny = [1e-200, 0.0]
        XCTAssertEqual(v.vectorProjection(onto: tiny), [0.0, 0.0])
        XCTAssertEqual(v.scalarProjection(onto: tiny), 0.0)
        XCTAssertEqual(v.orthogonalComponent(to: tiny), v)
    }

    // Just above the underflow boundary the reference is a usable direction
    // and both methods compute a real projection, confirming the convention
    // path is not swallowing ordinary small-magnitude input.
    func testProjectionsComputeNormallyJustAboveUnderflow() {
        let v = [3.0, 4.0]
        let small = [1e-150, 0.0]
        XCTAssertEqual(v.scalarProjection(onto: small), 3.0, accuracy: 1e-12)
        XCTAssertEqual(v.vectorProjection(onto: small)[0], 3.0, accuracy: 1e-12)
    }

    // MARK: - Distance Tests

    func testDistance() {
        // Classic 3-4-5 triangle
        let v1 = [0.0, 0.0]
        let v2 = [3.0, 4.0]
        XCTAssertEqual(v1.distance(to: v2), 5.0)

        // Symmetric
        let pointA = [1.0, 2.0, 3.0]
        let pointB = [4.0, 6.0, 8.0]
        XCTAssertEqual(pointA.distance(to: pointB), pointB.distance(to: pointA), accuracy: 1e-10)

        // Distance to self is zero
        XCTAssertEqual(pointA.distance(to: pointA), 0.0)
    }

    // MARK: - Distance Metric Tests
    // Cross-validated against scipy.spatial.distance (euclidean/cityblock/cosine), scipy 1.17.1

    func testDistanceMetricEuclideanMatchesExisting() {
        // The .euclidean case must be indistinguishable from distance(to:)
        let pairs: [([Double], [Double])] = [
            ([1.0, 2.0], [4.0, 6.0]),
            ([-1.0, 2.5, -3.0], [4.0, -1.5, 0.5]),
            ([0.2, 1.7, 3.14, 9.9], [5.5, 0.0, 2.71, 1.1])
        ]
        for (a, b) in pairs {
            XCTAssertEqual(a.distance(to: b, metric: .euclidean), a.distance(to: b))
        }
    }

    func testDistanceMetricManhattan() {
        let a = [1.0, 2.0]
        let b = [4.0, 6.0]
        XCTAssertEqual(a.distance(to: b, metric: .manhattan), 7.0)

        // Cross-validated: scipy cityblock = 12.5
        let c = [-1.0, 2.5, -3.0]
        let d = [4.0, -1.5, 0.5]
        XCTAssertEqual(c.distance(to: d, metric: .manhattan), 12.5, accuracy: 1e-12)

        // Cross-validated: scipy cityblock = 16.23
        let e = [0.2, 1.7, 3.14, 9.9]
        let f = [5.5, 0.0, 2.71, 1.1]
        XCTAssertEqual(e.distance(to: f, metric: .manhattan), 16.23, accuracy: 1e-12)
    }

    func testDistanceMetricChebyshev() {
        // Cross-validated: scipy chebyshev = 4.0
        let a = [1.0, 2.0]
        let b = [4.0, 6.0]
        XCTAssertEqual(a.distance(to: b, metric: .chebyshev), 4.0, accuracy: 1e-12)

        // Cross-validated: scipy chebyshev = 5.0 (the |−3.0 − 0.5| = 3.5 axis loses
        // to the |−1.0 − 4.0| = 5.0 axis)
        let c = [-1.0, 2.5, -3.0]
        let d = [4.0, -1.5, 0.5]
        XCTAssertEqual(c.distance(to: d, metric: .chebyshev), 5.0, accuracy: 1e-12)

        // A point has zero distance to itself.
        XCTAssertEqual(a.distance(to: a, metric: .chebyshev), 0.0)
    }

    func testDistanceMetricSquaredEuclidean() {
        // Cross-validated: scipy sqeuclidean = 25.0 (= euclidean 5.0, squared)
        let a = [1.0, 2.0]
        let b = [4.0, 6.0]
        XCTAssertEqual(a.distance(to: b, metric: .squaredEuclidean), 25.0, accuracy: 1e-12)

        // Cross-validated: scipy sqeuclidean = 53.25
        let c = [-1.0, 2.5, -3.0]
        let d = [4.0, -1.5, 0.5]
        XCTAssertEqual(c.distance(to: d, metric: .squaredEuclidean), 53.25, accuracy: 1e-12)

        // Squared Euclidean is exactly Euclidean squared.
        let e = a.distance(to: b, metric: .euclidean)
        XCTAssertEqual(a.distance(to: b, metric: .squaredEuclidean), e * e, accuracy: 1e-12)
    }

    func testDistanceMetricMinkowski() {
        // Cross-validated: scipy minkowski(a, b, 3) = 4.497941445275415
        let a = [1.0, 2.0]
        let b = [4.0, 6.0]
        XCTAssertEqual(a.distance(to: b, metric: .minkowski(p: 3)), 4.497941445275415, accuracy: 1e-12)

        // Cross-validated: scipy minkowski(c, d, 3) = 6.143529891036942
        let c = [-1.0, 2.5, -3.0]
        let d = [4.0, -1.5, 0.5]
        XCTAssertEqual(c.distance(to: d, metric: .minkowski(p: 3)), 6.143529891036942, accuracy: 1e-12)
    }

    func testMinkowskiUnifiesTheFamily() {
        // The teaching guarantee: Minkowski generalizes the named metrics.
        let c = [-1.0, 2.5, -3.0]
        let d = [4.0, -1.5, 0.5]

        // p = 1 is Manhattan.
        XCTAssertEqual(c.distance(to: d, metric: .minkowski(p: 1)),
                       c.distance(to: d, metric: .manhattan), accuracy: 1e-12)

        // p = 2 is Euclidean.
        XCTAssertEqual(c.distance(to: d, metric: .minkowski(p: 2)),
                       c.distance(to: d, metric: .euclidean), accuracy: 1e-12)

        // As p grows, Minkowski approaches Chebyshev (scipy p=10 = 5.063996...).
        let chebyshev = c.distance(to: d, metric: .chebyshev)
        let largeP = c.distance(to: d, metric: .minkowski(p: 10))
        XCTAssertEqual(largeP, 5.063996220548665, accuracy: 1e-12)
        XCTAssertLessThan(largeP - chebyshev, 0.1)  // converging toward 5.0 from above
    }

    func testDistanceMetricCosine() {
        // Cross-validated: scipy cosine = 0.00772212328633226
        let a = [1.0, 2.0]
        let b = [4.0, 6.0]
        XCTAssertEqual(a.distance(to: b, metric: .cosine), 0.00772212328633226, accuracy: 1e-12)

        // Cross-validated: scipy cosine = 1.53349356567384 (obtuse angle exceeds 1.0)
        let c = [-1.0, 2.5, -3.0]
        let d = [4.0, -1.5, 0.5]
        XCTAssertEqual(c.distance(to: d, metric: .cosine), 1.53349356567384, accuracy: 1e-12)

        // Cross-validated: scipy cosine = 0.687368131699077
        let e = [0.2, 1.7, 3.14, 9.9]
        let f = [5.5, 0.0, 2.71, 1.1]
        XCTAssertEqual(e.distance(to: f, metric: .cosine), 0.687368131699077, accuracy: 1e-12)
    }

    func testDistanceMetricIdenticalVectors() {
        // Every metric reports zero distance from a vector to itself
        let v = [2.0, 3.0]
        XCTAssertEqual(v.distance(to: v, metric: .euclidean), 0.0)
        XCTAssertEqual(v.distance(to: v, metric: .manhattan), 0.0)
        XCTAssertEqual(v.distance(to: v, metric: .cosine), 0.0, accuracy: 1e-12)
    }

    func testDistanceMetricZeroVectorCosine() {
        // Zero magnitude leaves direction undefined; the convention is
        // distance 1.0, matching KNearestNeighbors' inline behavior
        let zero = [0.0, 0.0]
        let v = [3.0, 4.0]
        XCTAssertEqual(zero.distance(to: v, metric: .cosine), 1.0)
        XCTAssertEqual(v.distance(to: zero, metric: .cosine), 1.0)
        XCTAssertEqual(zero.distance(to: zero, metric: .cosine), 1.0)
    }

    func testDistanceMetricMatchesNearestNeighbors() {
        // Equivalence with KNearestNeighbors' internal metric switch: points
        // chosen so Euclidean and Manhattan disagree about the nearest
        // neighbor, and the k=1 prediction must track the public method
        let features = [[3.0, 3.0], [0.0, 4.5]]
        let labels = [0, 1]
        let query = [0.0, 0.0]

        // Euclidean: √18 ≈ 4.24 beats 4.5, so label 0 is nearest
        XCTAssertLessThan(query.distance(to: features[0], metric: .euclidean),
                          query.distance(to: features[1], metric: .euclidean))
        let euclidean = KNearestNeighbors.fit(features: features, labels: labels, k: 1, metric: .euclidean)
        XCTAssertEqual(euclidean.predict([query]), [0])

        // Manhattan: 6.0 loses to 4.5, so label 1 is nearest
        XCTAssertGreaterThan(query.distance(to: features[0], metric: .manhattan),
                             query.distance(to: features[1], metric: .manhattan))
        let manhattan = KNearestNeighbors.fit(features: features, labels: labels, k: 1, metric: .manhattan)
        XCTAssertEqual(manhattan.predict([query]), [1])

        // Cosine: the distant-but-colinear point beats the near-but-rotated one
        let directional = [[10.0, 1.0], [0.1, 1.0]]
        let directionalQuery = [1.0, 0.1]
        XCTAssertLessThan(directionalQuery.distance(to: directional[0], metric: .cosine),
                          directionalQuery.distance(to: directional[1], metric: .cosine))
        let cosine = KNearestNeighbors.fit(features: directional, labels: labels, k: 1, metric: .cosine)
        XCTAssertEqual(cosine.predict([directionalQuery]), [0])
    }

    // MARK: - Matrix Transformation Tests

    func testMatrixTransformation() {
        let v = [1.0, 2.0]
        let matrix = [[0.0, -1.0], [1.0, 0.0]]  // 90° rotation
        let result = v.transformedBy(matrix)
        XCTAssertEqual(result, [-2.0, 1.0])
    }

    func testMatrixTransform() {
        // 90° rotation
        let rotationMatrix = [[0.0, -1.0], [1.0, 0.0]]
        let vector = [1.0, 0.0]
        let result = rotationMatrix.transform(vector)
        XCTAssertEqual(result, [0.0, 1.0])

        // Verify transform() matches transformedBy()
        let matrix = [[2.0, 0.0], [0.0, 3.0]]
        let v = [4.0, 5.0]
        XCTAssertEqual(matrix.transform(v), v.transformedBy(matrix))

        // 3D transformation
        let matrix3D = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0]
        ]
        XCTAssertEqual(matrix3D.transform([1.0, 2.0, 3.0]), [1.0, 2.0, 6.0])
    }

    // MARK: - Averaged Tests

    // Covers basic averaging and edge cases
    func testAveraged() {
        let vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        XCTAssertEqual(vectors.averaged(), [4.0, 5.0, 6.0])

        // Edge cases
        XCTAssertNil(([[Double]]()).averaged())
        XCTAssertNil([[1.0, 2.0, 3.0], [4.0, 5.0]].averaged())
        XCTAssertEqual([[3.0, 4.0, 5.0]].averaged(), [3.0, 4.0, 5.0])
    }

    // MARK: - Column Extraction Tests

    // Covers square and rectangular matrix column extraction
    func testColumnExtraction() {
        let square = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        XCTAssertEqual(square.column(at: 0), [1, 4, 7])
        XCTAssertEqual(square.column(at: 1), [2, 5, 8])
        XCTAssertEqual(square.column(at: 2), [3, 6, 9])

        // Rectangular matrix
        let rect = [[1, 2, 3, 4], [5, 6, 7, 8]]
        XCTAssertEqual(rect.column(at: 2), [3, 7])
    }

    // MARK: - Transpose Tests

    func testTransposed() {
        let matrix = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]

        let transposed = matrix.transposed()

        XCTAssertEqual(transposed, [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0]
        ])

        // Verify transposed() and transpose() produce same result
        let square = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        XCTAssertEqual(square.transpose(), square.transposed())
    }

    // MARK: - Matrix Multiplication Tests

    // Covers 2x2, 3x3, non-square, identity, and integer matrix multiplication
    func testMatmulShapes() {
        // 2x2
        let a = [[1.0, 2.0], [3.0, 4.0]]
        let b = [[5.0, 6.0], [7.0, 8.0]]
        XCTAssertEqual(a.multiplyMatrix(b), [[19.0, 22.0], [43.0, 50.0]])

        // Identity preserves the input matrix
        let identity = [[1.0, 0.0], [0.0, 1.0]]
        let matrix = [[3.0, 4.0], [5.0, 6.0]]
        XCTAssertEqual(identity.multiplyMatrix(matrix), matrix)

        // (2×3) × (3×2) → (2×2)
        let nonSquareA = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let nonSquareB = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        XCTAssertEqual(nonSquareA.multiplyMatrix(nonSquareB), [[58.0, 64.0], [139.0, 154.0]])

        // 3x3
        let a3 = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        let b3 = [
            [9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0]
        ]
        XCTAssertEqual(a3.multiplyMatrix(b3), [
            [30.0, 24.0, 18.0],
            [84.0, 69.0, 54.0],
            [138.0, 114.0, 90.0]
        ])

        // Integer matrices
        let intA = [[1, 2], [3, 4]]
        let intB = [[5, 6], [7, 8]]
        XCTAssertEqual(intA.multiplyMatrix(intB), [[19, 22], [43, 50]])
    }

    // Matrix multiplication is non-commutative and composes transformations
    func testMatmulProperties() {
        // Non-commutative
        let a = [[1.0, 2.0], [3.0, 4.0]]
        let b = [[5.0, 6.0], [7.0, 8.0]]
        XCTAssertNotEqual(a.multiplyMatrix(b), b.multiplyMatrix(a))

        // Two 90° rotations = 180° rotation
        let rotate90 = [[0.0, -1.0], [1.0, 0.0]]
        let composed = rotate90.multiplyMatrix(rotate90)
        XCTAssertEqual(composed[0][0], -1.0, accuracy: 1e-10)
        XCTAssertEqual(composed[0][1], 0.0, accuracy: 1e-10)
        XCTAssertEqual(composed[1][0], 0.0, accuracy: 1e-10)
        XCTAssertEqual(composed[1][1], -1.0, accuracy: 1e-10)

        // Composing transformations matches sequential application
        let rotate = [[0.707, -0.707], [0.707, 0.707]]
        let scale = [[2.0, 0.0], [0.0, 3.0]]
        let combined = scale.multiplyMatrix(rotate)
        let vector = [1.0, 0.0]
        let result = combined.transform(vector)
        let step1 = rotate.transform(vector)
        let step2 = scale.transform(step1)
        XCTAssertEqual(result[0], step2[0], accuracy: 1e-10)
        XCTAssertEqual(result[1], step2[1], accuracy: 1e-10)
    }

    // MARK: - TopIndices Tests

    // Covers basic ordering, edge cases, duplicates, and label overload
    func testTopIndices() {
        // Basic — descending by score with index preserved
        let scores = [0.3, 0.9, 0.1, 0.7, 0.5]
        let top3 = scores.topIndices(k: 3)
        XCTAssertEqual(top3.count, 3)
        XCTAssertEqual(top3[0].index, 1)
        XCTAssertEqual(top3[0].score, 0.9)
        XCTAssertEqual(top3[1].index, 3)
        XCTAssertEqual(top3[1].score, 0.7)
        XCTAssertEqual(top3[2].index, 4)
        XCTAssertEqual(top3[2].score, 0.5)

        // Requesting more than available returns the full array
        XCTAssertEqual([0.3, 0.9].topIndices(k: 5).count, 2)

        // Empty array returns no entries
        XCTAssertEqual([Double]().topIndices(k: 3).count, 0)

        // Duplicates preserved in order
        let dups = [0.5, 0.9, 0.5, 0.9, 0.3]
        let topDups = dups.topIndices(k: 3)
        XCTAssertEqual(topDups[0].score, 0.9)
        XCTAssertEqual(topDups[1].score, 0.9)

        // Label overload pairs scores with labels
        let words = ["the", "cat", "dog", "sat"]
        let labeled = [0.3, 0.9, 0.1, 0.7].topIndices(k: 2, labels: words)
        XCTAssertEqual(labeled.count, 2)
        XCTAssertEqual(labeled[0].label, "cat")
        XCTAssertEqual(labeled[0].score, 0.9, accuracy: 0.001)
        XCTAssertEqual(labeled[1].label, "sat")
        XCTAssertEqual(labeled[1].score, 0.7, accuracy: 0.001)
    }

    // MARK: - FindDuplicates Tests

    // Covers single match, no matches, multiple pairs, and threshold sensitivity
    func testFindDuplicates() {
        // Exact duplicate gets flagged
        let docs = [
            [0.8, 0.6, 0.9],
            [0.8, 0.6, 0.9],
            [0.1, 0.2, 0.1]
        ]
        let single = docs.findDuplicates(threshold: 0.95)
        XCTAssertEqual(single.count, 1)
        XCTAssertEqual(single[0].index1, 0)
        XCTAssertEqual(single[0].index2, 1)
        XCTAssertEqual(single[0].similarity, 1.0, accuracy: 1e-10)

        // Orthogonal vectors produce no duplicates
        let orthogonal = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        XCTAssertEqual(orthogonal.findDuplicates(threshold: 0.95).count, 0)

        // Multiple duplicate pairs are all reported
        let multi = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        let multiResult = multi.findDuplicates(threshold: 0.99)
        XCTAssertEqual(multiResult.count, 2)
        let pairs = [(multiResult[0].index1, multiResult[0].index2), (multiResult[1].index1, multiResult[1].index2)]
        XCTAssertTrue(pairs.contains { $0 == (0, 1) })
        XCTAssertTrue(pairs.contains { $0 == (2, 3) })

        // Threshold sensitivity
        let close = [[0.8, 0.6], [0.7, 0.7]]
        XCTAssertEqual(close.findDuplicates(threshold: 0.99).count, 0)
        XCTAssertEqual(close.findDuplicates(threshold: 0.90).count, 1)
    }

    // MARK: - ClusterCohesion Tests

    // Covers perfect cohesion, low cohesion, and edge cases
    func testClusterCohesion() {
        // Perfect cohesion — identical vectors
        let perfect = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ]
        XCTAssertEqual(perfect.clusterCohesion(), 1.0, accuracy: 1e-10)

        // Low cohesion — orthogonal vectors
        let orthogonal = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        XCTAssertEqual(orthogonal.clusterCohesion(), 0.0, accuracy: 1e-10)
    }

    func testClusterCohesionEdgeCases() {
        // Single vector
        XCTAssertEqual([[0.5, 0.5, 0.5]].clusterCohesion(), 0.0)

        // Empty array
        XCTAssertEqual(([[Double]]()).clusterCohesion(), 0.0)

        // Two vectors — should equal cosine similarity between them
        let cluster = [[0.8, 0.6], [0.6, 0.8]]
        let expected = cluster[0].cosineOfAngle(with: cluster[1])
        XCTAssertEqual(cluster.clusterCohesion(), expected, accuracy: 1e-10)
    }

    // MARK: - Sum Tests

    func testSum() {
        XCTAssertEqual([1.0, 2.0, 3.0, 4.0].sum(), 10.0)
        XCTAssertEqual([5, 10, 15, 20].sum(), 50)
        XCTAssertEqual([-5.0, 3.0, -2.0, 8.0].sum(), 4.0)
        XCTAssertEqual([Double]().sum(), 0.0)
    }

    // MARK: - SortedIndices Tests

    // Covers basic ordering, mapping back to values, duplicates, empty, and already-sorted input
    func testSortedIndices() {
        // Basic ordering
        XCTAssertEqual([40.0, 10.0, 30.0, 20.0].sortedIndices(), [1, 3, 2, 0])

        // Mapping indices back to values produces the sorted array
        let values = [5.0, 2.0, 8.0, 1.0]
        XCTAssertEqual(values.sortedIndices().map { values[$0] }, [1.0, 2.0, 5.0, 8.0])

        // Duplicates preserved
        let dups = [3.0, 1.0, 3.0, 2.0]
        XCTAssertEqual(dups.sortedIndices().map { dups[$0] }, [1.0, 2.0, 3.0, 3.0])

        // Empty array
        XCTAssertEqual([Double]().sortedIndices(), [])

        // Already sorted
        XCTAssertEqual([1.0, 2.0, 3.0, 4.0].sortedIndices(), [0, 1, 2, 3])
    }

    // MARK: - Matrix Determinant Tests

    func testDeterminant() {
        // 1x1
        XCTAssertEqual([[5.0]].determinant, 5.0)

        // 2x2
        let m2 = [[3.0, 8.0], [4.0, 6.0]]
        XCTAssertEqual(m2.determinant, -14.0)

        // 2x2 identity
        XCTAssertEqual([[1.0, 0.0], [0.0, 1.0]].determinant, 1.0)

        // 3x3
        let m3 = [
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0]
        ]
        XCTAssertEqual(m3.determinant, 1.0, accuracy: 1e-10)

        // Singular
        let singular = [[1.0, 2.0], [2.0, 4.0]]
        XCTAssertEqual(singular.determinant, 0.0, accuracy: 1e-10)
    }

    // MARK: - Matrix Inverse Tests

    // Verify A * A^-1 = I across 2x2 and 3x3 matrices
    func testInverted() throws {
        // 2x2
        let m2 = [[4.0, 7.0], [2.0, 6.0]]
        let inv2 = try m2.inverted()
        let id2 = m2.multiplyMatrix(inv2)
        XCTAssertEqual(id2[0][0], 1.0, accuracy: 1e-10)
        XCTAssertEqual(id2[0][1], 0.0, accuracy: 1e-10)
        XCTAssertEqual(id2[1][0], 0.0, accuracy: 1e-10)
        XCTAssertEqual(id2[1][1], 1.0, accuracy: 1e-10)

        // 3x3
        let m3 = [
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0]
        ]
        let inv3 = try m3.inverted()
        let id3 = m3.multiplyMatrix(inv3)
        for i in 0..<3 {
            for j in 0..<3 {
                XCTAssertEqual(id3[i][j], i == j ? 1.0 : 0.0, accuracy: 1e-10)
            }
        }
    }

    func testInvertedErrorCases() {
        // Singular matrix
        let singular = [[1.0, 2.0], [2.0, 4.0]]
        XCTAssertThrowsError(try singular.inverted()) { error in
            XCTAssertEqual(error as? MatrixError, .singular)
        }

        // Non-square matrix
        let nonSquare = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        XCTAssertThrowsError(try nonSquare.inverted()) { error in
            XCTAssertEqual(error as? MatrixError, .notSquare)
        }
    }

    // MARK: - LogDeterminant Tests

    // Covers log determinant values, edge cases, and consistency with determinant
    func testLogDeterminant() {
        // 2x2
        let m2 = [[4.0, 3.0], [6.0, 3.0]]
        let ld2 = m2.logDeterminant
        XCTAssertEqual(ld2.sign, -1.0)
        XCTAssertEqual(ld2.logAbsValue, log(6.0), accuracy: 1e-10)
        XCTAssertEqual(ld2.value, -6.0, accuracy: 1e-10)

        // Identity
        let identity = [[1.0, 0.0], [0.0, 1.0]]
        let ldI = identity.logDeterminant
        XCTAssertEqual(ldI.sign, 1.0)
        XCTAssertEqual(ldI.logAbsValue, 0.0, accuracy: 1e-10)

        // Singular
        let singular = [[1.0, 2.0], [2.0, 4.0]]
        let ldS = singular.logDeterminant
        XCTAssertEqual(ldS.sign, 0.0)
        XCTAssertTrue(ldS.logAbsValue.isInfinite && ldS.logAbsValue < 0)

        // Consistency with determinant across several matrices
        let matrices: [[[Double]]] = [
            [[3.0, 8.0], [4.0, 6.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]],
            [[4.0, 7.0], [2.0, 6.0]]
        ]
        for matrix in matrices {
            XCTAssertEqual(matrix.logDeterminant.value, matrix.determinant, accuracy: 1e-10)
        }
    }

    // MARK: - Condition Number Tests

    // Covers identity, diagonal, singular, and scale invariance
    func testConditionNumber() {
        // Identity has condition number 1
        XCTAssertEqual([[1.0, 0.0], [0.0, 1.0]].conditionNumber, 1.0, accuracy: 1e-10)

        // Diagonal matrix — ratio of largest to smallest eigenvalue
        XCTAssertEqual([[2.0, 0.0], [0.0, 8.0]].conditionNumber, 4.0, accuracy: 1e-10)

        // Singular matrix has infinite condition number
        XCTAssertTrue([[1.0, 2.0], [2.0, 4.0]].conditionNumber.isInfinite)

        // Scaling by a constant should not change condition number
        let matrix = [[4.0, 1.0], [1.0, 3.0]]
        let scaled = [[8.0, 2.0], [2.0, 6.0]]
        XCTAssertEqual(matrix.conditionNumber, scaled.conditionNumber, accuracy: 1e-10)
    }
}
