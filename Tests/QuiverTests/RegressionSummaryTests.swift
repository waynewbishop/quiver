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

/// Tests for `LinearRegression.summary()` and the `RegressionSummary` typed return.
///
/// All expected values are cross-validated against `numpy + scipy.stats` running
/// the canonical OLS closed-form (X'X)^-1 X'y plus the t-distribution for p-values
/// — the same routine `statsmodels.OLS` runs internally. Reference script:
/// `/tmp/quiver_stats_ref/gen_ols_ref.py` (generated 2026-05-07).
final class RegressionSummaryTests: XCTestCase {

    // MARK: - Dataset 1: simple 1-feature regression with known closed form

    func testSummaryDataset1SimpleSlopeAndIntercept() throws {
        // numpy reference (n=5, df_resid=3, p=2):
        //   beta = [0.13, 1.95]
        //   se   = [0.12556538801, 0.03785938897]
        //   t    = [1.03531715274, 51.50637802005]
        //   p    = [0.37665501432, 1.6117537e-05]
        //   R²   = 0.99887044236629
        //   adjR² = 0.99849392315506
        //   resid SE = 0.11972189997379
        //   95% CI[0] = (-0.26960510515, 0.52960510515)
        //   95% CI[1] = ( 1.82951452745, 2.07048547255)
        let X: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let y: [Double] = [2.1, 3.9, 6.1, 8.0, 9.8]

        let model = try LinearRegression.fit(features: X, targets: y)
        let summary = try model.summary(features: X, targets: y)

        XCTAssertEqual(summary.n, 5)
        XCTAssertEqual(summary.degreesOfFreedom, 3)
        XCTAssertEqual(summary.coefficients.count, 2)

        XCTAssertEqual(summary.coefficients[0], 0.13, accuracy: 1e-9)
        XCTAssertEqual(summary.coefficients[1], 1.95, accuracy: 1e-9)

        XCTAssertEqual(summary.standardErrors[0], 0.12556538801224887, accuracy: 1e-9)
        XCTAssertEqual(summary.standardErrors[1], 0.03785938897200176, accuracy: 1e-9)

        XCTAssertEqual(summary.tStatistics[0], 1.0353171527437250, accuracy: 1e-7)
        XCTAssertEqual(summary.tStatistics[1], 51.506378020049112, accuracy: 1e-6)

        XCTAssertEqual(summary.pValues[0], 0.37665501431525295, accuracy: 1e-7)
        XCTAssertEqual(summary.pValues[1], 1.6117537475546229e-05, accuracy: 1e-9)

        XCTAssertEqual(summary.rSquared, 0.99887044236629186, accuracy: 1e-9)
        XCTAssertEqual(summary.adjustedRSquared, 0.99849392315505581, accuracy: 1e-9)
        XCTAssertEqual(summary.residualStandardError, 0.11972189997378624, accuracy: 1e-9)

        XCTAssertEqual(summary.confidenceIntervals[0].lower, -0.26960510515109404, accuracy: 1e-7)
        XCTAssertEqual(summary.confidenceIntervals[0].upper,  0.52960510515109915, accuracy: 1e-7)
        XCTAssertEqual(summary.confidenceIntervals[1].lower,  1.82951452744575540, accuracy: 1e-7)
        XCTAssertEqual(summary.confidenceIntervals[1].upper,  2.07048547255424700, accuracy: 1e-7)

        XCTAssertEqual(summary.confidenceLevel, 0.95)
    }

    // MARK: - Dataset 2: multi-feature regression with no collinearity

    func testSummaryDataset2MultiFeatureMatchesStatsmodels() throws {
        // Reference computed via numpy OLS (= statsmodels OLS internals).
        // Generated 2026-05-07 with seed 42, n=20, p=4 (intercept + 3 features).
        // Tolerance is 1e-6 across SEs / t-stats / CIs — well past the spec's 1e-6.
        let features: [[Double]] = [
            [5.993428306022466, 9.723471397657631, -1.704622923798615],
            [8.046059712816051, 9.531693250553328, -3.468273913898361],
            [8.158425631014783, 11.534869458305817, -3.9389487718699043],
            [6.085120087171929, 9.073164614375075, -3.931459507140514],
            [5.483924543132068, 6.173439510684404, -6.4498356650260655],
            [3.8754249415180544, 7.974337759331153, -2.371505334809452],
            [3.1839518489575775, 7.175392597329417, -0.06870246215689191],
            [4.548447399026928, 10.135056409375848, -5.849496372426914],
            [3.9112345509496347, 10.221845179419732, -5.301987154844605],
            [5.751396036691344, 8.79872262016239, -3.5833874995865536],
            [3.796586775541206, 13.704556369017876, -3.026994449475868],
            [2.8845781420882, 11.645089824206378, -5.4416872999420445],
            [5.4177271900095105, 6.080659752240448, -5.6563720977968615],
            [5.393722471738247, 11.476933159990821, -2.657263437620059],
            [4.768703435223519, 9.397792608821423, -5.957043980734855],
            [3.5603115832105825, 9.078722458080424, -0.8857555475621686],
            [5.687236579136923, 6.473919689274532, -2.35183206121041],
            [4.229835439167367, 8.646155999388082, -1.7766474223182642],
            [7.061999044991902, 11.862560238232398, -4.678435046445277],
            [4.381575248297571, 10.662526862807129, -1.0489097457552816],
        ]
        let targets: [Double] = [
            3.271665022897626, 2.328621570328447, -0.3205972697360693,
            -3.2588239663055574, -9.562114752959165, -1.1727149867724127,
            4.0380949519896046, -11.517356074968797, -11.513596939820596,
            -2.4692916127607303, -6.659390179099923, -14.109432244519937,
            -7.691904809046762, -0.64049012147209, -12.84249392821311,
            1.8349465468820592, 2.625540664124436, 0.15714693645296654,
            -4.296706780200477, 0.7913733706252859,
        ]

        let model = try LinearRegression.fit(features: features, targets: targets)
        let summary = try model.summary(features: features, targets: targets)

        XCTAssertEqual(summary.n, 20)
        XCTAssertEqual(summary.degreesOfFreedom, 16)
        XCTAssertEqual(summary.coefficients.count, 4)

        // numpy reference: beta = [1.94905642, 1.87084263, -0.49267500, 2.96407384]
        let expCoef = [
            1.9490564244291164,
            1.8708426317443800,
            -0.49267499941770537,
            2.96407384065670290,
        ]
        let expSE = [
            0.76463752860102596,
            0.08966946625887301,
            0.06391123230453694,
            0.07042504255337038,
        ]
        let expT = [
            2.5489939370293433,
            20.86376455440601,
            -7.7087388500054210,
            42.08835001285844,
        ]
        let expP = [
            0.021446834252828006,
            4.9804604884684522e-13,
            8.9419236237731070e-07,
            0.0,
        ]
        let expCILow = [
            0.32809727556435786,
            1.6807518550438536,
            -0.62816075945985383,
            2.81477941974993050,
        ]
        let expCIHigh = [
            3.5700155732938752,
            2.0609334084449062,
            -0.35718923937555691,
            3.11336826156347520,
        ]

        for i in 0..<4 {
            XCTAssertEqual(summary.coefficients[i], expCoef[i], accuracy: 1e-9, "coef[\(i)]")
            XCTAssertEqual(summary.standardErrors[i], expSE[i], accuracy: 1e-9, "se[\(i)]")
            XCTAssertEqual(summary.tStatistics[i], expT[i], accuracy: 1e-6, "t[\(i)]")
            XCTAssertEqual(summary.confidenceIntervals[i].lower, expCILow[i], accuracy: 1e-6, "ciLo[\(i)]")
            XCTAssertEqual(summary.confidenceIntervals[i].upper, expCIHigh[i], accuracy: 1e-6, "ciHi[\(i)]")
            // p[3] is computed as 0.0 by numpy (tCDF saturates) — accept anything < 1e-12 there.
            if expP[i] < 1e-12 {
                XCTAssertLessThan(summary.pValues[i], 1e-9, "p[\(i)] should be ~0")
            } else {
                let tol = Swift.max(1e-9, expP[i] * 1e-5)
                XCTAssertEqual(summary.pValues[i], expP[i], accuracy: tol, "p[\(i)]")
            }
        }

        XCTAssertEqual(summary.rSquared, 0.99205532837839916, accuracy: 1e-9)
        XCTAssertEqual(summary.adjustedRSquared, 0.99056570244934905, accuracy: 1e-9)
        XCTAssertEqual(summary.residualStandardError, 0.5691925528442091, accuracy: 1e-9)
    }

    // MARK: - Dataset 3: singular case — must throw, not silently return garbage

    func testSummarySingularThrows() throws {
        // Two identical features make X'X singular. The fit() throws first;
        // we call summary() on a model fitted from a different (non-singular)
        // dataset and pass the singular features to verify summary's own check.
        // First show that `fit` itself throws on the singular data:
        let Xsingular: [[Double]] = [
            [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],
        ]
        let ySingular: [Double] = [1.5, 3.0, 4.5, 6.0, 7.5]
        XCTAssertThrowsError(
            try LinearRegression.fit(features: Xsingular, targets: ySingular)
        ) { error in
            XCTAssertEqual(error as? MatrixError, MatrixError.singular)
        }

        // Even when a caller has a working model fitted on different data and
        // then calls `summary` against singular features, the inversion step
        // throws. The two columns below are linearly independent (different
        // slope and curvature pattern), so the model itself fits cleanly.
        let goodModel = try LinearRegression.fit(
            features: [[1.0, 1.0], [2.0, 4.0], [3.0, 9.0], [4.0, 16.0], [5.0, 25.0]],
            targets: [1.5, 3.0, 4.5, 6.0, 7.5]
        )
        // Summary against singular features must throw — Flag 2 invariant.
        XCTAssertThrowsError(
            try goodModel.summary(features: Xsingular, targets: ySingular)
        ) { error in
            XCTAssertEqual(error as? MatrixError, MatrixError.singular)
        }
    }

    // MARK: - Dataset 4: near-singular case — finite, inflated SEs

    func testSummaryNearSingularReportsInflatedSEs() throws {
        // Two features with correlation 0.9999. Determinant of X'X is small
        // but non-zero — the inversion succeeds and the standard errors
        // visibly inflate (≈ 6.0 each instead of typical sub-unit values).
        // numpy reference values match this implementation to ~1e-6.
        let features: [[Double]] = [
            [1.690525703800356, 1.6798556211042537],
            [-0.4659373705408328, -0.4507241379308314],
            [0.0328201636785844, 0.009467730502481609],
            [0.40751628299650783, 0.41504745663706666],
            [-0.7889230286257386, -0.8180386405622714],
            [0.00206557290594813, -0.007298746725822827],
            [-0.0008903858579313628, -0.017920111435270742],
            [-1.7547243063454208, -1.7338738932090485],
            [1.0176580056634934, 1.0425329020919545],
            [0.6004985159195495, 0.5957799685788866],
            [-0.6254289739667597, -0.6134769647833962],
            [-0.17154826119572117, -0.17407643483094123],
            [0.5052993741967516, 0.5132822516698785],
            [-0.261356415191647, -0.2719767391130371],
            [-0.2427490786725466, -0.26688376447053236],
            [-1.4532414124907906, -1.4785951166211118],
            [0.5545803118918878, 0.5599428795929444],
            [0.12388090528703843, 0.1556535166057523],
            [0.2744599237599636, 0.27824243833225387],
            [-1.5265245318698402, -1.5337907236149941],
            [1.6506996911864755, 1.6775739752743801],
            [0.15433553545635803, 0.15767597290179555],
            [-0.3871399432863881, -0.38566677197911925],
            [2.029072220761112, 2.0324412128299625],
            [-0.04538602986064608, -0.047253540736147376],
            [-1.4506786991465745, -1.4549101782398257],
            [-0.40522785542768675, -0.4254802732513735],
            [-2.288315101971723, -2.2809924114324924],
            [1.0493965493432547, 1.047951315923],
            [-0.41647431852001854, -0.3995603099884625],
        ]
        let targets: [Double] = [
            5.207312256072892, 0.09972708704309907, 2.0391874445858393,
            3.6572700407257974, 0.2595579932192994, 1.5685670349332927,
            1.418452992732385, -2.0553074313649007, 3.860344246498444,
            2.615724532450659, 1.4860096006819088, 1.5181140332060874,
            3.0569400566677416, 2.2070272539049745, 2.290295788661348,
            -0.987495604360041, 3.269785163826968, 2.583502491444043,
            2.448709602772747, -1.934789298473781, 5.601877950136601,
            2.752506785877545, 1.4319982492689602, 5.593003121096765,
            1.8130817950002263, -1.1925108286314665, 1.0599410883043994,
            -1.9355102981193404, 4.865028148803342, 1.4848464511332375,
        ]
        let model = try LinearRegression.fit(features: features, targets: targets)
        let summary = try model.summary(features: features, targets: targets)

        // numpy reference: SEs are inflated to ~6.0 (vs ~0.1 for an uncorrelated
        // problem at this n). We assert this honesty rather than swallowing it.
        XCTAssertGreaterThan(summary.standardErrors[1], 1.0,
            "near-singular case must report inflated SEs, not silent garbage")
        XCTAssertGreaterThan(summary.standardErrors[2], 1.0,
            "near-singular case must report inflated SEs, not silent garbage")

        // Match numpy values to ~1e-6.
        XCTAssertEqual(summary.coefficients[0],  2.0179408663783565, accuracy: 1e-9)
        XCTAssertEqual(summary.coefficients[1], -0.84455677071127866, accuracy: 1e-7)
        XCTAssertEqual(summary.coefficients[2],  2.8728332907012373, accuracy: 1e-7)

        XCTAssertEqual(summary.standardErrors[0], 0.09442256061403842, accuracy: 1e-7)
        XCTAssertEqual(summary.standardErrors[1], 6.111549956635475,   accuracy: 1e-5)
        XCTAssertEqual(summary.standardErrors[2], 6.092804732834900,   accuracy: 1e-5)

        XCTAssertEqual(summary.rSquared, 0.94480140561487702, accuracy: 1e-9)
        XCTAssertEqual(summary.adjustedRSquared, 0.94071262084560869, accuracy: 1e-9)
        XCTAssertEqual(summary.residualStandardError, 0.5157696410965618, accuracy: 1e-9)
    }

    // MARK: - typed-summary protocol conformances

    func testRegressionSummaryEquatable() throws {
        let X: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let y: [Double] = [2.1, 3.9, 6.1, 8.0, 9.8]
        let model = try LinearRegression.fit(features: X, targets: y)
        let a = try model.summary(features: X, targets: y)
        let b = try model.summary(features: X, targets: y)
        XCTAssertEqual(a, b)
    }

    func testRegressionSummaryCodableRoundTrip() throws {
        let X: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let y: [Double] = [2.1, 3.9, 6.1, 8.0, 9.8]
        let model = try LinearRegression.fit(features: X, targets: y)
        let summary = try model.summary(features: X, targets: y)

        let data = try JSONEncoder().encode(summary)
        let decoded = try JSONDecoder().decode(RegressionSummary.self, from: data)
        XCTAssertEqual(summary, decoded)
    }

    func testRegressionSummaryDescriptionContainsKeyFields() throws {
        let X: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let y: [Double] = [2.1, 3.9, 6.1, 8.0, 9.8]
        let model = try LinearRegression.fit(features: X, targets: y)
        let summary = try model.summary(features: X, targets: y)

        let text = summary.description
        XCTAssertTrue(text.contains("Linear Regression Summary"))
        XCTAssertTrue(text.contains("R²"))
        XCTAssertTrue(text.contains("std err"))
    }

    func testRegressionSummaryMarkdownTableContainsHeaderAndRows() throws {
        let X: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let y: [Double] = [2.1, 3.9, 6.1, 8.0, 9.8]
        let model = try LinearRegression.fit(features: X, targets: y)
        let summary = try model.summary(features: X, targets: y)

        let md = summary.markdownTable()
        XCTAssertTrue(md.contains("| Term |"))
        XCTAssertTrue(md.contains("Coef"))
        XCTAssertTrue(md.contains("**R²**"))
        // Header + separator + 2 coef rows + summary line = 5 non-empty lines.
        let lines = md.split(separator: "\n", omittingEmptySubsequences: true)
        XCTAssertEqual(lines.count, 5)
    }

    func testRegressionSummaryCsvRowsHasExpectedHeader() throws {
        let X: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let y: [Double] = [2.1, 3.9, 6.1, 8.0, 9.8]
        let model = try LinearRegression.fit(features: X, targets: y)
        let summary = try model.summary(features: X, targets: y)

        let csv = summary.csvRows()
        let lines = csv.split(separator: "\n")
        XCTAssertEqual(lines.first, "term,coef,se,t,p,ci_lower,ci_upper")
        // Header + 2 coef rows = 3.
        XCTAssertEqual(lines.count, 3)
    }

    // MARK: - confidence-level configurability

    func testSummaryRespectsConfidenceLevel() throws {
        let X: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let y: [Double] = [2.1, 3.9, 6.1, 8.0, 9.8]
        let model = try LinearRegression.fit(features: X, targets: y)

        let s95 = try model.summary(features: X, targets: y, level: 0.95)
        let s99 = try model.summary(features: X, targets: y, level: 0.99)

        XCTAssertEqual(s95.confidenceLevel, 0.95)
        XCTAssertEqual(s99.confidenceLevel, 0.99)

        // 99% interval is wider than 95% interval at every coefficient.
        for i in 0..<s95.coefficients.count {
            let width95 = s95.confidenceIntervals[i].upper - s95.confidenceIntervals[i].lower
            let width99 = s99.confidenceIntervals[i].upper - s99.confidenceIntervals[i].lower
            XCTAssertGreaterThan(width99, width95, "coef \(i) — 99% CI must be wider")
        }
    }

    // MARK: - intercept-free model

    func testSummaryWithoutIntercept() throws {
        // y = 2x exactly, no intercept needed. n=5, p=1, df=4.
        let X: [[Double]] = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        let y: [Double] = [2.0, 4.0, 6.0, 8.0, 10.0]

        let model = try LinearRegression.fit(features: X, targets: y, intercept: false)
        let summary = try model.summary(features: X, targets: y)

        XCTAssertEqual(summary.n, 5)
        XCTAssertEqual(summary.degreesOfFreedom, 4)  // n - p, p = 1
        XCTAssertEqual(summary.coefficients.count, 1)
        XCTAssertEqual(summary.coefficients[0], 2.0, accuracy: 1e-12)
        // Residual SE is 0 (perfect fit) — t/p degenerate but should be defined.
        XCTAssertEqual(summary.residualStandardError, 0.0, accuracy: 1e-12)
        XCTAssertEqual(summary.rSquared, 1.0, accuracy: 1e-12)
    }
}
