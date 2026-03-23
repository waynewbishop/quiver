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

final class KMeansTests: XCTestCase {

    // Two well-separated clusters should be perfectly identified
    func testTwoClusters() {
        let data: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
            [8.0, 8.0], [8.5, 7.5], [9.0, 8.5]
        ]

        let model = KMeans.fit(data: data, k: 2, seed: 42)

        // All points in the first group should share a label
        let label0 = model.labels[0]
        XCTAssertEqual(model.labels[1], label0)
        XCTAssertEqual(model.labels[2], label0)

        // All points in the second group should share a different label
        let label1 = model.labels[3]
        XCTAssertEqual(model.labels[4], label1)
        XCTAssertEqual(model.labels[5], label1)

        // The two groups should have different labels
        XCTAssertNotEqual(label0, label1)
    }

    // Model should store correct number of centroids
    func testCentroidCount() {
        let data: [[Double]] = [
            [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
            [10.0, 10.0], [11.0, 11.0], [12.0, 12.0],
            [20.0, 20.0], [21.0, 21.0], [22.0, 22.0]
        ]

        let model = KMeans.fit(data: data, k: 3, seed: 7)
        XCTAssertEqual(model.centroids.count, 3)
        XCTAssertEqual(model.featureCount, 2)
    }

    // Centroids should be near the center of well-separated clusters
    func testCentroidAccuracy() {
        let data: [[Double]] = [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [10.0, 10.0], [11.0, 10.0], [10.0, 11.0]
        ]

        let model = KMeans.fit(data: data, k: 2, seed: 42)

        // Sort centroids by first coordinate to make assertions stable
        let sorted = model.centroids.sorted { $0[0] < $1[0] }

        // Low cluster centroid should be near (0.33, 0.33)
        XCTAssertEqual(sorted[0][0], 1.0 / 3.0, accuracy: 0.01)
        XCTAssertEqual(sorted[0][1], 1.0 / 3.0, accuracy: 0.01)

        // High cluster centroid should be near (10.33, 10.33)
        XCTAssertEqual(sorted[1][0], 31.0 / 3.0, accuracy: 0.01)
        XCTAssertEqual(sorted[1][1], 31.0 / 3.0, accuracy: 0.01)
    }

    // Inertia should be zero when k equals the number of unique points
    func testInertiaWithKEqualsSamples() {
        let data: [[Double]] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        let model = KMeans.fit(data: data, k: 3, seed: 42)
        XCTAssertEqual(model.inertia, 0.0, accuracy: 1e-9)
    }

    // Predict should assign new points to the nearest centroid
    func testPredict() {
        let data: [[Double]] = [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [10.0, 10.0], [11.0, 10.0], [10.0, 11.0]
        ]

        let model = KMeans.fit(data: data, k: 2, seed: 42)

        // Point near the low cluster
        let pred1 = model.predict([[0.5, 0.5]])
        XCTAssertEqual(pred1[0], model.labels[0])

        // Point near the high cluster
        let pred2 = model.predict([[10.5, 10.5]])
        XCTAssertEqual(pred2[0], model.labels[3])
    }

    // Reproducible results with the same seed
    func testSeedReproducibility() {
        let data: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
            [6.0, 9.0], [3.0, 4.0], [7.0, 7.5]
        ]

        let model1 = KMeans.fit(data: data, k: 2, seed: 99)
        let model2 = KMeans.fit(data: data, k: 2, seed: 99)
        XCTAssertEqual(model1.labels, model2.labels)
        XCTAssertEqual(model1.centroids, model2.centroids)
        XCTAssertEqual(model1.inertia, model2.inertia)
    }

    // Elbow method returns one inertia value per k
    func testElbowMethod() {
        let data: [[Double]] = [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [10.0, 10.0], [11.0, 10.0], [10.0, 11.0]
        ]

        let kRange = Array(1...4)
        let inertias = KMeans.elbowMethod(data: data, kRange: kRange, seed: 42)

        // One result per k value
        XCTAssertEqual(inertias.count, 4)

        // Inertia should decrease as k increases
        XCTAssertGreaterThan(inertias[0], inertias[1])

        // k=2 should capture the structure well (big drop from k=1)
        XCTAssertGreaterThan(inertias[0] - inertias[1], inertias[1] - inertias[2])
    }

    // Clusters method groups data by label and provides Sequence conformance
    func testClustersFromData() {
        let data: [[Double]] = [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [10.0, 10.0], [11.0, 10.0], [10.0, 11.0]
        ]

        let model = KMeans.fit(data: data, k: 2, seed: 42)
        let clusters = model.clusters(from: data)

        // One cluster per centroid
        XCTAssertEqual(clusters.count, 2)

        // Total points across all clusters equals input count
        let totalPoints = clusters.reduce(0) { $0 + $1.count }
        XCTAssertEqual(totalPoints, data.count)

        // Each cluster should have 3 points
        let sortedClusters = clusters.sorted { $0.centroid[0] < $1.centroid[0] }
        XCTAssertEqual(sortedClusters[0].count, 3)
        XCTAssertEqual(sortedClusters[1].count, 3)

        // Sequence conformance — can iterate over points
        var iteratedCount = 0
        for point in sortedClusters[0] {
            XCTAssertEqual(point.count, 2)
            iteratedCount += 1
        }
        XCTAssertEqual(iteratedCount, 3)
    }

    // Single cluster should assign all points the same label
    func testSingleCluster() {
        let data: [[Double]] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        let model = KMeans.fit(data: data, k: 1, seed: 42)
        XCTAssertTrue(model.labels.allSatisfy { $0 == 0 })
        XCTAssertEqual(model.centroids.count, 1)
        XCTAssertEqual(model.centroids[0][0], 3.0, accuracy: 1e-9)
        XCTAssertEqual(model.centroids[0][1], 4.0, accuracy: 1e-9)
    }

    // MARK: - Equatable

    // Cluster supports == comparison
    func testClusterEquatable() {
        let data: [[Double]] = [
            [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],
            [8.0, 8.0], [8.5, 7.5], [9.0, 8.5]
        ]
        let model = KMeans.fit(data: data, k: 2, seed: 42)

        let clusters1 = model.clusters(from: data)
        let clusters2 = model.clusters(from: data)
        XCTAssertEqual(clusters1, clusters2)
    }
}
