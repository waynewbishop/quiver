import XCTest
@testable import Quiver

final class QuartilesTests: XCTestCase {

    func testStructuredAccessMatchesExpectedValues() throws {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let q = try XCTUnwrap(data.quartiles())

        XCTAssertEqual(q.min, 1.0)
        XCTAssertEqual(q.q1, 3.25, accuracy: 1e-10)
        XCTAssertEqual(q.median, 5.5, accuracy: 1e-10)
        XCTAssertEqual(q.q3, 7.75, accuracy: 1e-10)
        XCTAssertEqual(q.max, 10.0)
        XCTAssertEqual(q.iqr, 4.5, accuracy: 1e-10)
    }

    func testReturnsNilOnEmptyArray() {
        let empty: [Double] = []
        XCTAssertNil(empty.quartiles())
    }

    func testDescriptionContainsAllLabels() throws {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0]
        let q = try XCTUnwrap(data.quartiles())
        let text = q.description

        XCTAssertTrue(text.contains("min:"))
        XCTAssertTrue(text.contains("q1:"))
        XCTAssertTrue(text.contains("median:"))
        XCTAssertTrue(text.contains("q3:"))
        XCTAssertTrue(text.contains("max:"))
        XCTAssertTrue(text.contains("iqr:"))
    }

    func testCodableRoundTrip() throws {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let original = try XCTUnwrap(data.quartiles())

        let encoded = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(Quartiles<Double>.self, from: encoded)

        XCTAssertEqual(original, restored)
    }

    func testEquatable() throws {
        let a = try XCTUnwrap([1.0, 2.0, 3.0, 4.0, 5.0].quartiles())
        let b = try XCTUnwrap([1.0, 2.0, 3.0, 4.0, 5.0].quartiles())
        let c = try XCTUnwrap([1.0, 2.0, 3.0, 4.0, 100.0].quartiles())

        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }
}
