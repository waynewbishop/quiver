import XCTest
@testable import Quiver

final class EmpiricalRuleTests: XCTestCase {

    func testNormalishDataMatchesExpectedFractions() throws {
        let data = [
            -2.5, -2.1, -1.8, -1.4, -1.1, -0.9, -0.7, -0.5, -0.3, -0.1,
             0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,
             1.0,  1.1,  1.2,  1.3,  1.4,  1.6,  1.8,  2.0,  2.3,  2.6
        ]
        let rule = try XCTUnwrap(data.empiricalRule())

        XCTAssertEqual(rule.count, 30)
        XCTAssertGreaterThanOrEqual(rule.within1Sigma, 0.55)
        XCTAssertLessThanOrEqual(rule.within1Sigma, 0.80)
        XCTAssertGreaterThanOrEqual(rule.within2Sigma, 0.85)
        XCTAssertGreaterThanOrEqual(rule.within3Sigma, 0.95)
    }

    func testExpectedValuesAreGaussianConstants() {
        let rule = EmpiricalRule(count: 0, within1Sigma: 0, within2Sigma: 0, within3Sigma: 0)
        XCTAssertEqual(rule.expected1Sigma, 0.6827)
        XCTAssertEqual(rule.expected2Sigma, 0.9545)
        XCTAssertEqual(rule.expected3Sigma, 0.9973)
    }

    func testReturnsNilOnEmptyArray() {
        let empty: [Double] = []
        XCTAssertNil(empty.empiricalRule())
    }

    func testReturnsNilOnSingleElement() {
        XCTAssertNil([3.0].empiricalRule())
    }

    func testReturnsNilOnZeroVariance() {
        XCTAssertNil([5.0, 5.0, 5.0, 5.0].empiricalRule())
    }

    func testReturnsNilOnNonFiniteValues() {
        XCTAssertNil([1.0, 2.0, .nan, 4.0].empiricalRule())
        XCTAssertNil([1.0, 2.0, .infinity, 4.0].empiricalRule())
    }

    func testAllValuesFallWithinThreeSigmaForSmallArray() throws {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0]
        let rule = try XCTUnwrap(data.empiricalRule())
        XCTAssertEqual(rule.within3Sigma, 1.0, accuracy: 1e-12)
    }

    func testDescriptionContainsExpectedLabels() throws {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0]
        let rule = try XCTUnwrap(data.empiricalRule())
        let text = rule.description

        XCTAssertTrue(text.contains("Empirical rule check"))
        XCTAssertTrue(text.contains("n = 5"))
        XCTAssertTrue(text.contains("within 1\u{03C3}"))
        XCTAssertTrue(text.contains("within 2\u{03C3}"))
        XCTAssertTrue(text.contains("within 3\u{03C3}"))
        XCTAssertTrue(text.contains("actual"))
        XCTAssertTrue(text.contains("expected"))
        XCTAssertTrue(text.contains("diff"))
    }

    func testCodableRoundTrip() throws {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let original = try XCTUnwrap(data.empiricalRule())

        let encoded = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(EmpiricalRule.self, from: encoded)

        XCTAssertEqual(original, restored)
    }

    func testEquatable() throws {
        let a = try XCTUnwrap([1.0, 2.0, 3.0, 4.0, 5.0].empiricalRule())
        let b = try XCTUnwrap([1.0, 2.0, 3.0, 4.0, 5.0].empiricalRule())
        let c = try XCTUnwrap([1.0, 2.0, 3.0, 4.0, 100.0].empiricalRule())

        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }
}
