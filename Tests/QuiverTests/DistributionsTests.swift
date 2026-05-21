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
        guard let v = Distributions.normal.cdf(x: 0, mean: 0, standardDeviation: 1) else {
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
            guard let actual = Distributions.normal.cdf(x: c.x, mean: 0, standardDeviation: 1) else {
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
            guard let actual = Distributions.normal.cdf(x: c.x, mean: 0, standardDeviation: 1) else {
                XCTFail("cdf returned nil for x=\(c.x)"); continue
            }
            let relErr = abs(actual - c.expected) / abs(c.expected)
            XCTAssertLessThan(relErr, 1e-7, "x=\(c.x), relErr=\(relErr)")
        }
    }

    func testNormalCDFAt196() {
        // Canonical 95% one-tailed value
        guard let actual = Distributions.normal.cdf(x: 1.96, mean: 0, standardDeviation: 1) else {
            XCTFail("cdf returned nil"); return
        }
        XCTAssertEqual(actual, 0.9750021048517795, accuracy: 1e-9)
    }

    // MARK: - Normal CDF — invariants

    func testNormalCDFMonotonic() {
        var prev: Double = -.infinity
        for k in 0..<20 {
            let x = -3.0 + Double(k) * 0.3
            guard let v = Distributions.normal.cdf(x: x, mean: 0, standardDeviation: 1) else {
                XCTFail("cdf returned nil for x=\(x)"); return
            }
            XCTAssertGreaterThan(v, prev, "monotonicity violated at x=\(x)")
            prev = v
        }
    }

    func testNormalCDFSymmetry() {
        for x in stride(from: 0.0, through: 4.0, by: 0.5) {
            guard let a = Distributions.normal.cdf(x: x, mean: 0, standardDeviation: 1),
                  let b = Distributions.normal.cdf(x: -x, mean: 0, standardDeviation: 1) else {
                XCTFail("cdf returned nil"); return
            }
            XCTAssertEqual(a + b, 1.0, accuracy: 1e-12, "symmetry at x=\(x)")
        }
    }

    func testNormalCDFNonStandard() {
        // For mean=100, std=15, cdf at the mean is 0.5
        guard let v = Distributions.normal.cdf(x: 100, mean: 100, standardDeviation: 15) else {
            XCTFail("cdf returned nil"); return
        }
        XCTAssertEqual(v, 0.5, accuracy: 1e-12)
    }

    func testNormalCDFInvalidStd() {
        XCTAssertNil(Distributions.normal.cdf(x: 0, mean: 0, standardDeviation: 0))
        XCTAssertNil(Distributions.normal.cdf(x: 0, mean: 0, standardDeviation: -1))
        XCTAssertNil(Distributions.normal.cdf(x: 0, mean: 0, standardDeviation: -1e-10))
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
            guard let actual = Distributions.normal.quantile(p: c.p, mean: 0, standardDeviation: 1) else {
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
            guard let actual = Distributions.normal.quantile(p: c.p, mean: 0, standardDeviation: 1) else {
                XCTFail("quantile returned nil for p=\(c.p)"); continue
            }
            XCTAssertEqual(actual, c.expected, accuracy: 5e-4, "p=\(c.p)")
        }
    }

    // MARK: - Normal Quantile — BSM formula-switch smoothness

    func testNormalQuantileBSMTransitionLower() {
        // Probe across the BSM formula switch region (around p ≈ 0.08)
        guard let q079 = Distributions.normal.quantile(p: 0.079, mean: 0, standardDeviation: 1),
              let q080 = Distributions.normal.quantile(p: 0.080, mean: 0, standardDeviation: 1),
              let q081 = Distributions.normal.quantile(p: 0.081, mean: 0, standardDeviation: 1) else {
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
        guard let q919 = Distributions.normal.quantile(p: 0.919, mean: 0, standardDeviation: 1),
              let q920 = Distributions.normal.quantile(p: 0.920, mean: 0, standardDeviation: 1),
              let q921 = Distributions.normal.quantile(p: 0.921, mean: 0, standardDeviation: 1) else {
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
        XCTAssertNil(Distributions.normal.quantile(p: 1e-15, mean: 0, standardDeviation: 1))
        XCTAssertNil(Distributions.normal.quantile(p: 1e-20, mean: 0, standardDeviation: 1))
        XCTAssertNil(Distributions.normal.quantile(p: 0.0, mean: 0, standardDeviation: 1))
        XCTAssertNil(Distributions.normal.quantile(p: -0.1, mean: 0, standardDeviation: 1))
        XCTAssertNil(Distributions.normal.quantile(p: 1.0, mean: 0, standardDeviation: 1))
        XCTAssertNil(Distributions.normal.quantile(p: 1.1, mean: 0, standardDeviation: 1))
    }

    func testNormalQuantileInvalidStd() {
        XCTAssertNil(Distributions.normal.quantile(p: 0.5, mean: 0, standardDeviation: 0))
        XCTAssertNil(Distributions.normal.quantile(p: 0.5, mean: 0, standardDeviation: -1))
    }

    // MARK: - Normal — round-trip cdf(quantile(p)) ≈ p

    func testNormalCDFQuantileRoundTrip() {
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            guard let q = Distributions.normal.quantile(p: p, mean: 0, standardDeviation: 1),
                  let backToP = Distributions.normal.cdf(x: q, mean: 0, standardDeviation: 1) else {
                XCTFail("nil in round trip for p=\(p)"); return
            }
            XCTAssertEqual(backToP, p, accuracy: 1e-6, "round trip at p=\(p)")
        }
    }

    func testNormalQuantileNonStandard() {
        // For mean=100, std=15, quantile at 0.5 is the mean
        guard let v = Distributions.normal.quantile(p: 0.5, mean: 100, standardDeviation: 15) else {
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
            guard let actual = Distributions.normal.pdf(x: c.x, mean: 0, standardDeviation: 1) else {
                XCTFail("pdf returned nil for x=\(c.x)"); continue
            }
            XCTAssertEqual(actual, c.expected, accuracy: 1e-12, "x=\(c.x)")
        }
    }

    func testNormalPDFSymmetry() {
        for x in stride(from: 0.5, through: 4.0, by: 0.5) {
            guard let a = Distributions.normal.pdf(x: x, mean: 0, standardDeviation: 1),
                  let b = Distributions.normal.pdf(x: -x, mean: 0, standardDeviation: 1) else {
                XCTFail("pdf returned nil"); return
            }
            XCTAssertEqual(a, b, accuracy: 1e-15, "pdf symmetry at x=\(x)")
        }
    }

    func testNormalPDFInvalidStd() {
        XCTAssertNil(Distributions.normal.pdf(x: 0, mean: 0, standardDeviation: 0))
        XCTAssertNil(Distributions.normal.pdf(x: 0, mean: 0, standardDeviation: -1))
    }

    // MARK: - Normal logPDF

    func testNormalLogPDFEqualsLogPDF() {
        for x in [-2.0, -0.5, 0.0, 0.5, 2.0] {
            guard let lp = Distributions.normal.logPDF(x: x, mean: 0, standardDeviation: 1),
                  let p = Distributions.normal.pdf(x: x, mean: 0, standardDeviation: 1) else {
                XCTFail("nil at x=\(x)"); return
            }
            XCTAssertEqual(lp, log(p), accuracy: 1e-12, "logPDF == log(pdf) at x=\(x)")
        }
    }

    func testNormalLogPDFRealisticSmallStd() {
        // Realistic small std (matches GaussianNaiveBayes' internal variance floor of 1e-9 → std ~3e-5).
        // At this scale, log-space arithmetic must stay finite.
        guard let v = Distributions.normal.logPDF(x: 0, mean: 0, standardDeviation: 1e-4) else {
            XCTFail("logPDF returned nil for std=1e-4"); return
        }
        XCTAssertTrue(v.isFinite, "logPDF should be finite at std=1e-4, got \(v)")
    }

    func testNormalLogPDFExtremeSmallStdReturnsNil() {
        // At std=1e-300 the computation underflows to NaN. Per the public API contract,
        // the function returns nil rather than silently propagating a non-finite result.
        XCTAssertNil(Distributions.normal.logPDF(x: 0, mean: 0, standardDeviation: 1e-300))
    }

    func testNormalPDFExtremeSmallStdReturnsNil() {
        XCTAssertNil(Distributions.normal.pdf(x: 0, mean: 0, standardDeviation: 1e-300))
    }

    func testNormalLogPDFLargeX() {
        // Large |x|: returns very negative finite value, no overflow
        guard let v = Distributions.normal.logPDF(x: 1000, mean: 0, standardDeviation: 1) else {
            XCTFail("logPDF returned nil for x=1000"); return
        }
        XCTAssertTrue(v.isFinite, "logPDF should be finite at x=1000, got \(v)")
        XCTAssertLessThan(v, -1000.0, "logPDF should be very negative at x=1000")
    }

    func testNormalLogPDFInvalidStd() {
        XCTAssertNil(Distributions.normal.logPDF(x: 0, mean: 0, standardDeviation: 0))
        XCTAssertNil(Distributions.normal.logPDF(x: 0, mean: 0, standardDeviation: -1))
    }

    // MARK: - t-distribution CDF — reference grid (scipy.stats.t.cdf)

    func testTCDFReferenceGrid() {
        // Reference values from scipy.stats.t.cdf, generated 2026-05-07.
        // Each row: (x, df, expected). Tolerance is 1e-9 — these values are
        // accurate to machine precision and the implementation should match.
        let cases: [(x: Double, df: Double, expected: Double)] = [
            // df = 1 (Cauchy)
            (x: 0.0,    df: 1,    expected: 5.0000000000000000e-01),
            (x: 1.0,    df: 1,    expected: 7.5000000000000000e-01),
            (x: -1.0,   df: 1,    expected: 2.5000000000000006e-01),
            (x: 2.0,    df: 1,    expected: 8.5241638234956674e-01),
            (x: 10.0,   df: 1,    expected: 9.6827448256944648e-01),
            (x: -10.0,  df: 1,    expected: 3.1725517430553567e-02),
            // df = 2
            (x: 0.0,    df: 2,    expected: 5.0000000000000000e-01),
            (x: 1.0,    df: 2,    expected: 7.8867513459481287e-01),
            (x: 2.0,    df: 2,    expected: 9.0824829046386302e-01),
            (x: 3.0,    df: 2,    expected: 9.5226701686664539e-01),
            (x: -3.0,   df: 2,    expected: 4.7732983133354563e-02),
            // df = 3
            (x: 0.0,    df: 3,    expected: 5.0000000000000000e-01),
            (x: 1.0,    df: 3,    expected: 8.0449889052211476e-01),
            (x: 2.0,    df: 3,    expected: 9.3033701572057848e-01),
            (x: 3.182,  df: 3,    expected: 9.7499143172834313e-01),
            // df = 5
            (x: 0.0,    df: 5,    expected: 5.0000000000000000e-01),
            (x: 1.0,    df: 5,    expected: 8.1839126617543867e-01),
            (x: 2.571,  df: 5,    expected: 9.7501268265807428e-01),
            (x: 4.0,    df: 5,    expected: 9.9483829225958431e-01),
            (x: -4.0,   df: 5,    expected: 5.1617077404157224e-03),
            // df = 10
            (x: 0.0,    df: 10,   expected: 5.0000000000000000e-01),
            (x: 2.228,  df: 10,   expected: 9.7499411409144432e-01),
            (x: 3.169,  df: 10,   expected: 9.9499768331780758e-01),
            (x: 5.0,    df: 10,   expected: 9.9973133319862173e-01),
            // df = 30
            (x: 0.0,    df: 30,   expected: 5.0000000000000000e-01),
            (x: 2.042,  df: 30,   expected: 9.7498566467190106e-01),
            (x: 1.96,   df: 30,   expected: 9.7032884355197480e-01),
            // df = 100
            (x: 0.0,    df: 100,  expected: 5.0000000000000000e-01),
            (x: 1.984,  df: 100,  expected: 9.7500161310191635e-01),
            // df = 1000
            (x: 0.0,    df: 1000, expected: 5.0000000000000000e-01),
            (x: 1.96,   df: 1000, expected: 9.7486340752212564e-01),
        ]
        for c in cases {
            guard let actual = Distributions.t.cdf(x: c.x, df: c.df) else {
                XCTFail("t.cdf returned nil for x=\(c.x), df=\(c.df)"); continue
            }
            XCTAssertEqual(
                actual, c.expected,
                accuracy: 1e-9,
                "t.cdf(x=\(c.x), df=\(c.df)) — got \(actual), expected \(c.expected)"
            )
        }
    }

    // MARK: - t-distribution CDF — symmetry invariant

    func testTCDFSymmetryInvariant() {
        // tCDF(-x, df) + tCDF(x, df) == 1, exactly, for every (x, df).
        // The implementation derives both sides from the same beta call so
        // the invariant should hold to 1e-12 (well past the spec's 1e-10).
        for df in [1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1000.0] {
            for x in stride(from: 0.1, through: 5.0, by: 0.4) {
                guard let pos = Distributions.t.cdf(x: x, df: df),
                      let neg = Distributions.t.cdf(x: -x, df: df) else {
                    XCTFail("t.cdf returned nil at x=\(x), df=\(df)"); return
                }
                XCTAssertEqual(
                    pos + neg, 1.0,
                    accuracy: 1e-12,
                    "symmetry violated at x=\(x), df=\(df) — pos+neg=\(pos + neg)"
                )
            }
        }
    }

    // MARK: - t-distribution CDF — converges to normal as df → ∞

    func testTCDFConvergesToNormal() {
        // At df = 1000 the t-CDF matches the normal CDF to ~1e-4.
        // This is the standard cross-check that verifies the regularized
        // incomplete beta is plumbed correctly through the t-distribution.
        for x in stride(from: -3.0, through: 3.0, by: 0.5) {
            guard let tValue = Distributions.t.cdf(x: x, df: 1000),
                  let nValue = Distributions.normal.cdf(x: x, mean: 0, standardDeviation: 1) else {
                XCTFail("cdf returned nil at x=\(x)"); return
            }
            XCTAssertEqual(
                tValue, nValue,
                accuracy: 1e-3,
                "t (df=1000) and normal differ at x=\(x): t=\(tValue), n=\(nValue)"
            )
        }
    }

    // MARK: - t-distribution CDF — edge cases

    func testTCDFCauchyClosedForm() {
        // The t-distribution with df=1 is the standard Cauchy:
        // tCDF(0, 1) == 0.5 exactly; tCDF(1, 1) == 0.75 exactly.
        guard let v0 = Distributions.t.cdf(x: 0, df: 1),
              let v1 = Distributions.t.cdf(x: 1, df: 1) else {
            XCTFail("t.cdf returned nil"); return
        }
        XCTAssertEqual(v0, 0.5, accuracy: 1e-15)
        XCTAssertEqual(v1, 0.75, accuracy: 1e-12)
    }

    func testTCDFInvalidDF() {
        XCTAssertNil(Distributions.t.cdf(x: 0, df: 0))
        XCTAssertNil(Distributions.t.cdf(x: 0, df: -1))
        XCTAssertNil(Distributions.t.cdf(x: 0, df: -1e-10))
    }

    func testTCDFMonotonic() {
        for df in [1.0, 5.0, 30.0] {
            var prev: Double = -.infinity
            for k in 0..<30 {
                let x = -4.0 + Double(k) * 0.3
                guard let v = Distributions.t.cdf(x: x, df: df) else {
                    XCTFail("t.cdf returned nil"); return
                }
                XCTAssertGreaterThan(v, prev, "monotonicity violated at x=\(x), df=\(df)")
                prev = v
            }
        }
    }

    // MARK: - t-distribution Quantile — reference grid

    func testTQuantileReferenceGrid() {
        // Reference values from scipy.stats.t.ppf, generated 2026-05-07.
        // Tolerance 5e-6 — bisection is converged to ~1e-12 internally,
        // but tests use a looser bound to be robust to any future rebracketing.
        let cases: [(p: Double, df: Double, expected: Double)] = [
            (p: 0.025, df: 1,    expected: -1.2706204736174705e+01),
            (p: 0.05,  df: 1,    expected: -6.3137515146750438e+00),
            (p: 0.95,  df: 1,    expected:  6.3137515146750367e+00),
            (p: 0.975, df: 1,    expected:  1.2706204736174694e+01),
            (p: 0.99,  df: 1,    expected:  3.1820515953773935e+01),
            (p: 0.5,   df: 1,    expected:  0.0),
            (p: 0.025, df: 2,    expected: -4.3026527297494637e+00),
            (p: 0.95,  df: 2,    expected:  2.9199855803537242e+00),
            (p: 0.975, df: 2,    expected:  4.3026527297494619e+00),
            (p: 0.025, df: 3,    expected: -3.1824463052837086e+00),
            (p: 0.975, df: 3,    expected:  3.1824463052837078e+00),
            (p: 0.025, df: 5,    expected: -2.5705818356363155e+00),
            (p: 0.95,  df: 5,    expected:  2.0150483733330229e+00),
            (p: 0.975, df: 5,    expected:  2.5705818356363146e+00),
            (p: 0.025, df: 10,   expected: -2.2281388519862748e+00),
            (p: 0.95,  df: 10,   expected:  1.8124611228116760e+00),
            (p: 0.975, df: 10,   expected:  2.2281388519862739e+00),
            (p: 0.025, df: 30,   expected: -2.0422724563012382e+00),
            (p: 0.975, df: 30,   expected:  2.0422724563012378e+00),
            (p: 0.975, df: 100,  expected:  1.9839715185235520e+00),
            (p: 0.975, df: 1000, expected:  1.9623390808264078e+00),
        ]
        for c in cases {
            guard let actual = Distributions.t.quantile(p: c.p, df: c.df) else {
                XCTFail("t.quantile returned nil for p=\(c.p), df=\(c.df)"); continue
            }
            // Use absolute or relative — whichever is looser.
            let tol = Swift.max(5e-6, abs(c.expected) * 5e-7)
            XCTAssertEqual(
                actual, c.expected,
                accuracy: tol,
                "t.quantile(p=\(c.p), df=\(c.df)) — got \(actual), expected \(c.expected)"
            )
        }
    }

    // MARK: - t-distribution Quantile — round trip

    func testTQuantileRoundTrip() {
        // quantile(cdf(x, df), df) ≈ x to ~1e-6 — bisection precision.
        for df in [2.0, 5.0, 30.0, 100.0] {
            for x in stride(from: -2.5, through: 2.5, by: 0.5) {
                guard let p = Distributions.t.cdf(x: x, df: df),
                      let recovered = Distributions.t.quantile(p: p, df: df) else {
                    XCTFail("round-trip nil at x=\(x), df=\(df)"); continue
                }
                XCTAssertEqual(
                    recovered, x,
                    accuracy: 1e-6,
                    "round-trip failed at x=\(x), df=\(df) — got \(recovered)"
                )
            }
        }
    }

    func testTQuantileMedianIsZero() {
        for df in [1.0, 5.0, 30.0, 1000.0] {
            guard let q = Distributions.t.quantile(p: 0.5, df: df) else {
                XCTFail("nil"); return
            }
            XCTAssertEqual(q, 0.0, accuracy: 1e-12, "median should be zero, df=\(df)")
        }
    }

    func testTQuantileInvalidInputs() {
        XCTAssertNil(Distributions.t.quantile(p: 0, df: 5))
        XCTAssertNil(Distributions.t.quantile(p: 1, df: 5))
        XCTAssertNil(Distributions.t.quantile(p: -0.1, df: 5))
        XCTAssertNil(Distributions.t.quantile(p: 1.1, df: 5))
        XCTAssertNil(Distributions.t.quantile(p: 0.5, df: 0))
        XCTAssertNil(Distributions.t.quantile(p: 0.5, df: -3))
    }

    // MARK: - chi-squared CDF — closed-form check at df=2

    func testChiSquaredCDFClosedFormAtDF2() {
        // At df = 2, chi² CDF is exactly 1 - exp(-x/2).
        // This is the most stringent accuracy check — should match to 1e-12.
        for x in stride(from: 0.1, through: 20.0, by: 0.5) {
            guard let actual = Distributions.chiSquared.cdf(x: x, df: 2) else {
                XCTFail("nil at x=\(x)"); continue
            }
            let expected = 1.0 - Foundation.exp(-x / 2.0)
            XCTAssertEqual(
                actual, expected,
                accuracy: 1e-12,
                "df=2 closed-form mismatch at x=\(x): got \(actual), expected \(expected)"
            )
        }
    }

    // MARK: - chi-squared CDF — reference grid (scipy.stats.chi2.cdf)

    func testChiSquaredCDFReferenceGrid() {
        // Reference values from scipy.stats.chi2.cdf, generated 2026-05-07.
        let cases: [(x: Double, df: Double, expected: Double)] = [
            (x: 0.001,  df: 1,  expected: 2.5227120630039609e-02),
            (x: 0.5,    df: 1,  expected: 5.2049987781304663e-01),
            (x: 1.0,    df: 1,  expected: 6.8268949213708585e-01),
            (x: 2.0,    df: 1,  expected: 8.4270079294971512e-01),
            (x: 3.841,  df: 1,  expected: 9.4998631623604335e-01),
            (x: 5.0,    df: 1,  expected: 9.7465268132253180e-01),
            (x: 10.0,   df: 1,  expected: 9.9843459774199750e-01),
            (x: 0.5,    df: 2,  expected: 2.2119921692859512e-01),
            (x: 1.0,    df: 2,  expected: 3.9346934028736652e-01),
            (x: 2.0,    df: 2,  expected: 6.3212055882855767e-01),
            (x: 5.991,  df: 2,  expected: 9.4998838497342086e-01),
            (x: 10.0,   df: 2,  expected: 9.9326205300091452e-01),
            (x: 1.0,    df: 5,  expected: 3.7434226752703623e-02),
            (x: 5.0,    df: 5,  expected: 5.8411981300449189e-01),
            (x: 11.07,  df: 5,  expected: 9.4999038137759451e-01),
            (x: 15.086, df: 5,  expected: 9.8999887523781416e-01),
            (x: 20.0,   df: 5,  expected: 9.9875026943696865e-01),
            (x: 3.0,    df: 10, expected: 1.8575936222140668e-02),
            (x: 10.0,   df: 10, expected: 5.5950671493478776e-01),
            (x: 18.307, df: 10, expected: 9.4999941090860196e-01),
            (x: 23.209, df: 10, expected: 9.8999913418525920e-01),
            (x: 30.0,   df: 10, expected: 9.9914335878922467e-01),
            (x: 15.0,   df: 30, expected: 1.0260427912342602e-02),
            (x: 43.773, df: 30, expected: 9.5000029221162674e-01),
            (x: 50.892, df: 30, expected: 9.8999955884203739e-01),
            (x: 60.0,   df: 30, expected: 9.9907931760385138e-01),
        ]
        for c in cases {
            guard let actual = Distributions.chiSquared.cdf(x: c.x, df: c.df) else {
                XCTFail("chi².cdf returned nil for x=\(c.x), df=\(c.df)"); continue
            }
            XCTAssertEqual(
                actual, c.expected,
                accuracy: 1e-9,
                "chi².cdf(x=\(c.x), df=\(c.df)) — got \(actual), expected \(c.expected)"
            )
        }
    }

    // MARK: - chi-squared CDF — boundary and edge cases

    func testChiSquaredCDFAtZero() {
        for df in [1.0, 5.0, 30.0] {
            guard let v = Distributions.chiSquared.cdf(x: 0, df: df) else {
                XCTFail("nil at df=\(df)"); return
            }
            XCTAssertEqual(v, 0.0, accuracy: 1e-15)
        }
    }

    func testChiSquaredCDFNegativeXReturnsZero() {
        for df in [1.0, 5.0, 30.0] {
            guard let v = Distributions.chiSquared.cdf(x: -1, df: df) else {
                XCTFail("nil"); return
            }
            XCTAssertEqual(v, 0.0)
        }
    }

    func testChiSquaredCDFFarTailApproachesOne() {
        // At x = 10 * df the CDF should be very close to 1 but still < 1.
        for df in [1.0, 5.0, 10.0, 30.0] {
            guard let v = Distributions.chiSquared.cdf(x: 10.0 * df, df: df) else {
                XCTFail("nil"); return
            }
            XCTAssertGreaterThan(v, 0.99, "df=\(df), x=\(10.0 * df), got \(v)")
            XCTAssertLessThanOrEqual(v, 1.0)
        }
    }

    func testChiSquaredCDFInvalidDF() {
        XCTAssertNil(Distributions.chiSquared.cdf(x: 1, df: 0))
        XCTAssertNil(Distributions.chiSquared.cdf(x: 1, df: -1))
        XCTAssertNil(Distributions.chiSquared.cdf(x: 1, df: -1e-10))
    }

    func testChiSquaredCDFMonotonic() {
        for df in [1.0, 5.0, 30.0] {
            var prev: Double = -.infinity
            for k in 1..<40 {
                let x = Double(k) * 0.5
                guard let v = Distributions.chiSquared.cdf(x: x, df: df) else {
                    XCTFail("nil"); return
                }
                XCTAssertGreaterThan(v, prev, "monotonicity violated at x=\(x), df=\(df)")
                prev = v
            }
        }
    }

    // MARK: - Poisson

    // Anchor values verified against scipy.stats.poisson
    func testPoissonPMFKnownValues() {
        // PMF(k=2, λ=3.5) = exp(-3.5) * 3.5^2 / 2! ≈ 0.184959
        guard let v = Distributions.poisson.pmf(k: 2, lambda: 3.5) else {
            XCTFail("nil"); return
        }
        XCTAssertEqual(v, 0.184959, accuracy: 1e-6)

        // PMF(k=0, λ=1.0) = exp(-1.0) ≈ 0.367879
        guard let v0 = Distributions.poisson.pmf(k: 0, lambda: 1.0) else {
            XCTFail("nil"); return
        }
        XCTAssertEqual(v0, Foundation.exp(-1.0), accuracy: 1e-12)
    }

    func testPoissonLogPMFMatchesLogOfPMF() {
        // log(pmf) and logPMF must agree across a range of parameters
        for lambda in [0.5, 1.0, 3.5, 10.0] {
            for k in 0..<15 {
                guard let pmf = Distributions.poisson.pmf(k: k, lambda: lambda),
                      let logPMF = Distributions.poisson.logPMF(k: k, lambda: lambda) else {
                    XCTFail("nil at λ=\(lambda), k=\(k)"); return
                }
                XCTAssertEqual(Foundation.log(pmf), logPMF, accuracy: 1e-10)
            }
        }
    }

    // PMF over the full support sums to 1 — partition invariant
    func testPoissonPMFSumsToOne() {
        for lambda in [0.5, 1.0, 3.5, 10.0] {
            var total = 0.0
            for k in 0..<100 {
                guard let v = Distributions.poisson.pmf(k: k, lambda: lambda) else {
                    XCTFail("nil"); return
                }
                total += v
            }
            XCTAssertEqual(total, 1.0, accuracy: 1e-10, "λ=\(lambda)")
        }
    }

    // Anchor verified against scipy.stats.poisson.cdf
    func testPoissonCDFKnownValues() {
        // CDF(k=5, λ=3.5) ≈ 0.857614
        guard let v = Distributions.poisson.cdf(k: 5, lambda: 3.5) else {
            XCTFail("nil"); return
        }
        XCTAssertEqual(v, 0.857614, accuracy: 1e-6)

        // CDF(k=0, λ=1.0) = exp(-1.0)
        guard let v0 = Distributions.poisson.cdf(k: 0, lambda: 1.0) else {
            XCTFail("nil"); return
        }
        XCTAssertEqual(v0, Foundation.exp(-1.0), accuracy: 1e-12)
    }

    // CDF is monotonically nondecreasing and bounded in [0, 1]
    func testPoissonCDFMonotonic() {
        for lambda in [0.5, 3.5, 10.0] {
            var prev = -1.0
            for k in 0..<30 {
                guard let v = Distributions.poisson.cdf(k: k, lambda: lambda) else {
                    XCTFail("nil"); return
                }
                XCTAssertGreaterThanOrEqual(v, prev)
                XCTAssertGreaterThanOrEqual(v, 0.0)
                XCTAssertLessThanOrEqual(v, 1.0)
                prev = v
            }
        }
    }

    // Quantile round-trips through the CDF: cdf(quantile(p)) >= p, cdf(quantile(p) - 1) < p
    func testPoissonQuantileRoundTrip() {
        for lambda in [0.5, 3.5, 10.0] {
            for p in [0.1, 0.25, 0.5, 0.75, 0.95, 0.99] {
                guard let k = Distributions.poisson.quantile(p: p, lambda: lambda),
                      let cdfAtK = Distributions.poisson.cdf(k: k, lambda: lambda) else {
                    XCTFail("nil at λ=\(lambda), p=\(p)"); return
                }
                XCTAssertGreaterThanOrEqual(cdfAtK, p, "λ=\(lambda), p=\(p), k=\(k)")
                if k > 0 {
                    guard let cdfAtKMinus1 = Distributions.poisson.cdf(k: k - 1, lambda: lambda) else {
                        XCTFail("nil"); return
                    }
                    XCTAssertLessThan(cdfAtKMinus1, p, "λ=\(lambda), p=\(p), k=\(k)")
                }
            }
        }
    }

    // 95th percentile of Poisson(3.5) is 7 per scipy.stats.poisson.ppf
    func testPoissonQuantileKnownValue() {
        XCTAssertEqual(Distributions.poisson.quantile(p: 0.95, lambda: 3.5), 7)
    }

    func testPoissonMeanAndVariance() {
        XCTAssertEqual(Distributions.poisson.mean(lambda: 3.5), 3.5)
        XCTAssertEqual(Distributions.poisson.variance(lambda: 3.5), 3.5)
    }

    func testPoissonInvalidInputs() {
        XCTAssertNil(Distributions.poisson.pmf(k: 2, lambda: 0))
        XCTAssertNil(Distributions.poisson.pmf(k: 2, lambda: -1))
        XCTAssertNil(Distributions.poisson.pmf(k: -1, lambda: 1))
        XCTAssertNil(Distributions.poisson.logPMF(k: 2, lambda: 0))
        XCTAssertNil(Distributions.poisson.cdf(k: 5, lambda: 0))
        XCTAssertNil(Distributions.poisson.quantile(p: 0.5, lambda: 0))
        XCTAssertNil(Distributions.poisson.quantile(p: -0.1, lambda: 1))
        XCTAssertNil(Distributions.poisson.quantile(p: 1.1, lambda: 1))
        XCTAssertNil(Distributions.poisson.mean(lambda: 0))
        XCTAssertNil(Distributions.poisson.variance(lambda: -1))
    }

    // CDF returns 0 for negative k
    func testPoissonCDFNegativeK() {
        XCTAssertEqual(Distributions.poisson.cdf(k: -1, lambda: 3.5), 0.0)
        XCTAssertEqual(Distributions.poisson.cdf(k: -100, lambda: 1.0), 0.0)
    }

    // MARK: - Binomial

    // Anchor values verified against scipy.stats.binom
    func testBinomialPMFKnownValues() {
        // PMF(k=3, n=10, p=0.4) = C(10,3) * 0.4^3 * 0.6^7 ≈ 0.214991
        guard let v = Distributions.binomial.pmf(k: 3, n: 10, p: 0.4) else {
            XCTFail("nil"); return
        }
        XCTAssertEqual(v, 0.214991, accuracy: 1e-6)

        // PMF(k=0, n=10, p=0.4) = 0.6^10
        guard let v0 = Distributions.binomial.pmf(k: 0, n: 10, p: 0.4) else {
            XCTFail("nil"); return
        }
        XCTAssertEqual(v0, Foundation.pow(0.6, 10), accuracy: 1e-12)

        // PMF(k=10, n=10, p=0.4) = 0.4^10
        guard let v10 = Distributions.binomial.pmf(k: 10, n: 10, p: 0.4) else {
            XCTFail("nil"); return
        }
        XCTAssertEqual(v10, Foundation.pow(0.4, 10), accuracy: 1e-12)
    }

    func testBinomialLogPMFMatchesLogOfPMF() {
        for (n, p) in [(10, 0.4), (20, 0.5), (50, 0.1)] {
            for k in 0...n {
                guard let pmf = Distributions.binomial.pmf(k: k, n: n, p: p),
                      let logPMF = Distributions.binomial.logPMF(k: k, n: n, p: p) else {
                    XCTFail("nil at n=\(n), k=\(k), p=\(p)"); return
                }
                // Skip log comparison when pmf rounds to zero — logPMF is -infinity there
                if pmf > 0 {
                    XCTAssertEqual(Foundation.log(pmf), logPMF, accuracy: 1e-9)
                }
            }
        }
    }

    // PMF over the full support {0, ..., n} sums to 1 — partition invariant
    func testBinomialPMFSumsToOne() {
        for (n, p) in [(10, 0.4), (20, 0.5), (50, 0.1), (100, 0.7)] {
            var total = 0.0
            for k in 0...n {
                guard let v = Distributions.binomial.pmf(k: k, n: n, p: p) else {
                    XCTFail("nil"); return
                }
                total += v
            }
            XCTAssertEqual(total, 1.0, accuracy: 1e-10, "n=\(n), p=\(p)")
        }
    }

    // Anchor verified against scipy.stats.binom.cdf
    func testBinomialCDFKnownValues() {
        // CDF(k=3, n=10, p=0.4) ≈ 0.382281
        guard let v = Distributions.binomial.cdf(k: 3, n: 10, p: 0.4) else {
            XCTFail("nil"); return
        }
        XCTAssertEqual(v, 0.382281, accuracy: 1e-6)

        // CDF at the support boundaries
        XCTAssertEqual(Distributions.binomial.cdf(k: 10, n: 10, p: 0.4), 1.0)
        XCTAssertEqual(Distributions.binomial.cdf(k: -1, n: 10, p: 0.4), 0.0)
    }

    // Boundary probabilities p = 0 and p = 1 short-circuit the beta evaluation
    func testBinomialCDFBoundaryProbabilities() {
        // p = 0: always 0 successes, so CDF(k>=0) = 1
        XCTAssertEqual(Distributions.binomial.cdf(k: 0, n: 10, p: 0.0), 1.0)
        XCTAssertEqual(Distributions.binomial.cdf(k: 5, n: 10, p: 0.0), 1.0)

        // p = 1: always n successes, so CDF(k<n) = 0, CDF(k>=n) = 1
        XCTAssertEqual(Distributions.binomial.cdf(k: 5, n: 10, p: 1.0), 0.0)
        XCTAssertEqual(Distributions.binomial.cdf(k: 10, n: 10, p: 1.0), 1.0)
    }

    // CDF is monotonically nondecreasing across k for fixed n, p
    func testBinomialCDFMonotonic() {
        for (n, p) in [(20, 0.3), (50, 0.5), (100, 0.7)] {
            var prev = -1.0
            for k in 0...n {
                guard let v = Distributions.binomial.cdf(k: k, n: n, p: p) else {
                    XCTFail("nil"); return
                }
                XCTAssertGreaterThanOrEqual(v, prev)
                XCTAssertGreaterThanOrEqual(v, 0.0)
                XCTAssertLessThanOrEqual(v, 1.0)
                prev = v
            }
        }
    }

    // Quantile round-trips through the CDF
    func testBinomialQuantileRoundTrip() {
        for (n, prob) in [(10, 0.4), (50, 0.5), (100, 0.3)] {
            for p in [0.1, 0.25, 0.5, 0.75, 0.95] {
                guard let k = Distributions.binomial.quantile(p: p, n: n, probability: prob),
                      let cdfAtK = Distributions.binomial.cdf(k: k, n: n, p: prob) else {
                    XCTFail("nil at n=\(n), p=\(p)"); return
                }
                XCTAssertGreaterThanOrEqual(cdfAtK, p, "n=\(n), p=\(p), k=\(k)")
                if k > 0 {
                    guard let cdfAtKMinus1 = Distributions.binomial.cdf(k: k - 1, n: n, p: prob) else {
                        XCTFail("nil"); return
                    }
                    XCTAssertLessThan(cdfAtKMinus1, p)
                }
            }
        }
    }

    // 95th percentile of Binomial(n=10, p=0.4) is 7 per scipy.stats.binom.ppf
    func testBinomialQuantileKnownValue() {
        XCTAssertEqual(Distributions.binomial.quantile(p: 0.95, n: 10, probability: 0.4), 7)
        XCTAssertEqual(Distributions.binomial.quantile(p: 0.5, n: 10, probability: 0.4), 4)
    }

    func testBinomialMeanAndVariance() {
        XCTAssertEqual(Distributions.binomial.mean(n: 10, p: 0.4), 4.0)

        guard let variance = Distributions.binomial.variance(n: 10, p: 0.4) else {
            XCTFail("nil"); return
        }
        XCTAssertEqual(variance, 2.4, accuracy: 1e-12)

        // Variance vanishes at boundary probabilities
        XCTAssertEqual(Distributions.binomial.variance(n: 10, p: 0.0), 0.0)
        XCTAssertEqual(Distributions.binomial.variance(n: 10, p: 1.0), 0.0)
    }

    func testBinomialInvalidInputs() {
        XCTAssertNil(Distributions.binomial.pmf(k: 5, n: 10, p: -0.1))
        XCTAssertNil(Distributions.binomial.pmf(k: 5, n: 10, p: 1.1))
        XCTAssertNil(Distributions.binomial.pmf(k: 11, n: 10, p: 0.5))
        XCTAssertNil(Distributions.binomial.pmf(k: -1, n: 10, p: 0.5))
        XCTAssertNil(Distributions.binomial.pmf(k: 5, n: -1, p: 0.5))
        XCTAssertNil(Distributions.binomial.cdf(k: 5, n: 10, p: -0.1))
        XCTAssertNil(Distributions.binomial.quantile(p: -0.1, n: 10, probability: 0.5))
        XCTAssertNil(Distributions.binomial.quantile(p: 0.5, n: 10, probability: -0.1))
        XCTAssertNil(Distributions.binomial.mean(n: -1, p: 0.5))
        XCTAssertNil(Distributions.binomial.variance(n: 10, p: 1.5))
    }
}
