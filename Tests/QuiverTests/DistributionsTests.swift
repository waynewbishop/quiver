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

final class DistributionsTests: XCTestCase {

    // MARK: - Normal CDF — body values

    func testNormalCDFAtZero() {
        guard let v = Distributions.normal.cdf(x: 0, mean: 0, std: 1) else {
            XCTFail("cdf returned nil"); return
        }
        XCTAssertEqual(v, 0.5, accuracy: 1e-12)
    }

    func testNormalCDFBodyGrid() {
        // Reference values from scipy.stats.norm.cdf
        let cases: [(x: Double, expected: Double)] = [
            (-4, 3.167124183311998e-05),
            (-2, 0.02275013194817921),
            (-1, 0.15865525393145707),
            ( 0, 0.5),
            ( 1, 0.8413447460685429),
            ( 2, 0.9772498680518208),
            ( 4, 0.9999683287581669),
        ]
        for c in cases {
            guard let actual = Distributions.normal.cdf(x: c.x, mean: 0, std: 1) else {
                XCTFail("cdf returned nil for x=\(c.x)"); continue
            }
            XCTAssertEqual(actual, c.expected, accuracy: 1e-9, "x=\(c.x)")
        }
    }

    func testNormalCDFTailGridRelativeError() {
        // At |x| ≥ 6 the CDF approaches 1 − O(10⁻⁹); use relative error.
        let cases: [(x: Double, expected: Double)] = [
            (-8, 6.106226635438361e-16),
            (-6, 9.865876449133282e-10),
            ( 6, 0.9999999990134123),
            ( 8, 0.9999999999999993),
        ]
        for c in cases {
            guard let actual = Distributions.normal.cdf(x: c.x, mean: 0, std: 1) else {
                XCTFail("cdf returned nil for x=\(c.x)"); continue
            }
            let relErr = abs(actual - c.expected) / abs(c.expected)
            XCTAssertLessThan(relErr, 1e-7, "x=\(c.x), relErr=\(relErr)")
        }
    }

    func testNormalCDFAt196() {
        // Canonical 95% one-tailed value
        guard let actual = Distributions.normal.cdf(x: 1.96, mean: 0, std: 1) else {
            XCTFail("cdf returned nil"); return
        }
        XCTAssertEqual(actual, 0.9750021048517795, accuracy: 1e-9)
    }

    // MARK: - Normal CDF — invariants

    func testNormalCDFMonotonic() {
        var prev: Double = -.infinity
        for k in 0..<20 {
            let x = -3.0 + Double(k) * 0.3
            guard let v = Distributions.normal.cdf(x: x, mean: 0, std: 1) else {
                XCTFail("cdf returned nil for x=\(x)"); return
            }
            XCTAssertGreaterThan(v, prev, "monotonicity violated at x=\(x)")
            prev = v
        }
    }

    func testNormalCDFSymmetry() {
        for x in stride(from: 0.0, through: 4.0, by: 0.5) {
            guard let a = Distributions.normal.cdf(x: x, mean: 0, std: 1),
                  let b = Distributions.normal.cdf(x: -x, mean: 0, std: 1) else {
                XCTFail("cdf returned nil"); return
            }
            XCTAssertEqual(a + b, 1.0, accuracy: 1e-12, "symmetry at x=\(x)")
        }
    }

    func testNormalCDFNonStandard() {
        // For mean=100, std=15, cdf at the mean is 0.5
        guard let v = Distributions.normal.cdf(x: 100, mean: 100, std: 15) else {
            XCTFail("cdf returned nil"); return
        }
        XCTAssertEqual(v, 0.5, accuracy: 1e-12)
    }

    func testNormalCDFInvalidStd() {
        XCTAssertNil(Distributions.normal.cdf(x: 0, mean: 0, std: 0))
        XCTAssertNil(Distributions.normal.cdf(x: 0, mean: 0, std: -1))
        XCTAssertNil(Distributions.normal.cdf(x: 0, mean: 0, std: -1e-10))
    }

    // MARK: - Normal Quantile — body values (5e-7 tolerance)

    func testNormalQuantileBody() {
        let cases: [(p: Double, expected: Double)] = [
            (0.001,  -3.0902323061678154),
            (0.0228, -1.9990772149717704),
            (0.5,     0.0),
            (0.9772,  1.99907721497177),
            (0.999,   3.090232306167797),
        ]
        for c in cases {
            guard let actual = Distributions.normal.quantile(p: c.p, mean: 0, std: 1) else {
                XCTFail("quantile returned nil for p=\(c.p)"); continue
            }
            XCTAssertEqual(actual, c.expected, accuracy: 5e-7, "p=\(c.p)")
        }
    }

    // MARK: - Normal Quantile — tail values (5e-4 tolerance)

    func testNormalQuantileTails() {
        let cases: [(p: Double, expected: Double)] = [
            (1e-9,    -5.997807010047333),
            (1e-6,    -4.753424308829469),
            (1 - 1e-6, 4.753424308829469),
            (1 - 1e-9, 5.997807010047333),
        ]
        for c in cases {
            guard let actual = Distributions.normal.quantile(p: c.p, mean: 0, std: 1) else {
                XCTFail("quantile returned nil for p=\(c.p)"); continue
            }
            XCTAssertEqual(actual, c.expected, accuracy: 5e-4, "p=\(c.p)")
        }
    }

    // MARK: - Normal Quantile — BSM formula-switch smoothness

    func testNormalQuantileBSMTransitionLower() {
        // Probe across the BSM formula switch region (around p ≈ 0.08)
        guard let q079 = Distributions.normal.quantile(p: 0.079, mean: 0, std: 1),
              let q080 = Distributions.normal.quantile(p: 0.080, mean: 0, std: 1),
              let q081 = Distributions.normal.quantile(p: 0.081, mean: 0, std: 1) else {
            XCTFail("quantile returned nil"); return
        }
        // Monotonic and approximately equally spaced — no visible step
        XCTAssertLessThan(q079, q080)
        XCTAssertLessThan(q080, q081)
        let step1 = q080 - q079
        let step2 = q081 - q080
        // Steps should be similar in size — wide tolerance to allow for second-derivative curvature
        XCTAssertEqual(step1, step2, accuracy: step1 * 0.1, "BSM transition not smooth around 0.08")
    }

    func testNormalQuantileBSMTransitionUpper() {
        guard let q919 = Distributions.normal.quantile(p: 0.919, mean: 0, std: 1),
              let q920 = Distributions.normal.quantile(p: 0.920, mean: 0, std: 1),
              let q921 = Distributions.normal.quantile(p: 0.921, mean: 0, std: 1) else {
            XCTFail("quantile returned nil"); return
        }
        XCTAssertLessThan(q919, q920)
        XCTAssertLessThan(q920, q921)
        let step1 = q920 - q919
        let step2 = q921 - q920
        XCTAssertEqual(step1, step2, accuracy: step1 * 0.1, "BSM transition not smooth around 0.92")
    }

    // MARK: - Normal Quantile — out of domain

    func testNormalQuantileOutOfDomain() {
        XCTAssertNil(Distributions.normal.quantile(p: 1e-15, mean: 0, std: 1))
        XCTAssertNil(Distributions.normal.quantile(p: 1e-20, mean: 0, std: 1))
        XCTAssertNil(Distributions.normal.quantile(p: 0.0, mean: 0, std: 1))
        XCTAssertNil(Distributions.normal.quantile(p: -0.1, mean: 0, std: 1))
        XCTAssertNil(Distributions.normal.quantile(p: 1.0, mean: 0, std: 1))
        XCTAssertNil(Distributions.normal.quantile(p: 1.1, mean: 0, std: 1))
    }

    func testNormalQuantileInvalidStd() {
        XCTAssertNil(Distributions.normal.quantile(p: 0.5, mean: 0, std: 0))
        XCTAssertNil(Distributions.normal.quantile(p: 0.5, mean: 0, std: -1))
    }

    // MARK: - Normal — round-trip cdf(quantile(p)) ≈ p

    func testNormalCDFQuantileRoundTrip() {
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            guard let q = Distributions.normal.quantile(p: p, mean: 0, std: 1),
                  let backToP = Distributions.normal.cdf(x: q, mean: 0, std: 1) else {
                XCTFail("nil in round trip for p=\(p)"); return
            }
            XCTAssertEqual(backToP, p, accuracy: 1e-6, "round trip at p=\(p)")
        }
    }

    func testNormalQuantileNonStandard() {
        // For mean=100, std=15, quantile at 0.5 is the mean
        guard let v = Distributions.normal.quantile(p: 0.5, mean: 100, std: 15) else {
            XCTFail("quantile returned nil"); return
        }
        XCTAssertEqual(v, 100.0, accuracy: 1e-9)
    }

    // MARK: - Normal PDF

    func testNormalPDFReferenceValues() {
        let cases: [(x: Double, expected: Double)] = [
            ( 0,    0.3989422804014327),
            ( 1,    0.24197072451914337),
            (-1,    0.24197072451914337),
            ( 1.96, 0.05844094433345173),
            ( 2,    0.05399096651318806),
        ]
        for c in cases {
            guard let actual = Distributions.normal.pdf(x: c.x, mean: 0, std: 1) else {
                XCTFail("pdf returned nil for x=\(c.x)"); continue
            }
            XCTAssertEqual(actual, c.expected, accuracy: 1e-12, "x=\(c.x)")
        }
    }

    func testNormalPDFSymmetry() {
        for x in stride(from: 0.5, through: 4.0, by: 0.5) {
            guard let a = Distributions.normal.pdf(x: x, mean: 0, std: 1),
                  let b = Distributions.normal.pdf(x: -x, mean: 0, std: 1) else {
                XCTFail("pdf returned nil"); return
            }
            XCTAssertEqual(a, b, accuracy: 1e-15, "pdf symmetry at x=\(x)")
        }
    }

    func testNormalPDFInvalidStd() {
        XCTAssertNil(Distributions.normal.pdf(x: 0, mean: 0, std: 0))
        XCTAssertNil(Distributions.normal.pdf(x: 0, mean: 0, std: -1))
    }

    // MARK: - Normal logPDF

    func testNormalLogPDFEqualsLogPDF() {
        for x in [-2.0, -0.5, 0.0, 0.5, 2.0] {
            guard let lp = Distributions.normal.logPDF(x: x, mean: 0, std: 1),
                  let p = Distributions.normal.pdf(x: x, mean: 0, std: 1) else {
                XCTFail("nil at x=\(x)"); return
            }
            XCTAssertEqual(lp, log(p), accuracy: 1e-12, "logPDF == log(pdf) at x=\(x)")
        }
    }

    func testNormalLogPDFRealisticSmallStd() {
        // Realistic small std (matches GaussianNaiveBayes' internal variance floor of 1e-9 → std ~3e-5).
        // At this scale, log-space arithmetic must stay finite.
        guard let v = Distributions.normal.logPDF(x: 0, mean: 0, std: 1e-4) else {
            XCTFail("logPDF returned nil for std=1e-4"); return
        }
        XCTAssertTrue(v.isFinite, "logPDF should be finite at std=1e-4, got \(v)")
    }

    func testNormalLogPDFExtremeSmallStdReturnsNil() {
        // At std=1e-300 the computation underflows to NaN. Per the public API contract,
        // the function returns nil rather than silently propagating a non-finite result.
        XCTAssertNil(Distributions.normal.logPDF(x: 0, mean: 0, std: 1e-300))
    }

    func testNormalPDFExtremeSmallStdReturnsNil() {
        XCTAssertNil(Distributions.normal.pdf(x: 0, mean: 0, std: 1e-300))
    }

    func testNormalLogPDFLargeX() {
        // Large |x|: returns very negative finite value, no overflow
        guard let v = Distributions.normal.logPDF(x: 1000, mean: 0, std: 1) else {
            XCTFail("logPDF returned nil for x=1000"); return
        }
        XCTAssertTrue(v.isFinite, "logPDF should be finite at x=1000, got \(v)")
        XCTAssertLessThan(v, -1000.0, "logPDF should be very negative at x=1000")
    }

    func testNormalLogPDFInvalidStd() {
        XCTAssertNil(Distributions.normal.logPDF(x: 0, mean: 0, std: 0))
        XCTAssertNil(Distributions.normal.logPDF(x: 0, mean: 0, std: -1))
    }
}
