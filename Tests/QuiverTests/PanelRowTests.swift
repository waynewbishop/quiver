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

final class PanelRowTests: XCTestCase {

    /// Builds a three-row run with columns out of alphabetical order.
    private func makeRun() -> Panel {
        Panel([
            ("time", [0.0, 5.0, 10.0]),
            ("heartRate", [142.0, 151.0, 158.0]),
            ("pace", [5.6, 5.2, 4.9])
        ])
    }

    // Iterating a panel yields one row per record, in row order
    func testIterationYieldsRowsInOrder() {
        var heartRates: [Double] = []
        var paces: [Double] = []
        for moment in makeRun() {
            heartRates.append(moment["heartRate"])
            paces.append(moment["pace"])
        }
        XCTAssertEqual(heartRates, [142.0, 151.0, 158.0])
        XCTAssertEqual(paces, [5.6, 5.2, 4.9])
    }

    // The number of rows iterated matches rowCount
    func testIterationCountMatchesRowCount() {
        let run = makeRun()
        var count = 0
        for _ in run {
            count += 1
        }
        XCTAssertEqual(count, run.rowCount)
        XCTAssertEqual(Array(run).count, 3)
    }

    // underestimatedCount reports the full row count
    func testUnderestimatedCountIsRowCount() {
        XCTAssertEqual(makeRun().underestimatedCount, 3)
    }

    // Each row keeps the panel's column order and values
    func testRowColumnNamesAndValues() {
        let first = Array(makeRun())[0]
        XCTAssertEqual(first.columnNames, ["time", "heartRate", "pace"])
        XCTAssertEqual(first.values, [0.0, 142.0, 5.6])
        XCTAssertEqual(first["time"], 0.0)
    }

    // A panel with a single row yields exactly one row
    func testSingleRowPanel() {
        let panel = Panel([("x", [7.0]), ("y", [8.0])])
        let rows = Array(panel)
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0]["y"], 8.0)
    }

    // A filter that matches nothing leaves a zero-row panel that yields no rows
    func testZeroRowPanelYieldsNoRows() {
        let empty = makeRun().filtered(where: [false, false, false])
        XCTAssertEqual(empty.rowCount, 0)
        var count = 0
        for _ in empty {
            count += 1
        }
        XCTAssertEqual(count, 0)
        XCTAssertEqual(empty.underestimatedCount, 0)
    }

    // Rows iterate after a filter, with only the kept records
    func testIterationAfterFilter() {
        let run = makeRun()
        let fast = run.filtered(where: run["pace"].isLessThan(5.5))
        var times: [Double] = []
        for moment in fast {
            times.append(moment["time"])
        }
        XCTAssertEqual(times, [5.0, 10.0])
    }

    // enumerated() pairs each row with its position
    func testEnumeratedGivesRowPositions() {
        var steepRows: [Int] = []
        for (index, moment) in makeRun().enumerated() where moment["heartRate"] > 150.0 {
            steepRows.append(index)
        }
        XCTAssertEqual(steepRows, [1, 2])
    }

    // Rows with the same names and values are equal; a different value is not
    func testEquatable() {
        let rows = Array(makeRun())
        let again = Array(makeRun())
        XCTAssertEqual(rows[0], again[0])
        XCTAssertNotEqual(rows[0], rows[1])
        XCTAssertTrue(makeRun().elementsEqual(makeRun()))
    }

    // description lists name and value pairs in column order
    func testDescription() {
        let first = Array(makeRun())[0]
        XCTAssertEqual(first.description, "Row: time 0.0, heartRate 142.0, pace 5.6")
    }

    // Existing Panel API still resolves alongside the Sequence members
    func testPanelAPIUnchangedBySequence() {
        let run = makeRun()
        XCTAssertEqual(run.filtered(where: [true, false, true]).rowCount, 2)
        XCTAssertEqual(run.sortedBy(column: "heartRate", ascending: false)["heartRate"], [158.0, 151.0, 142.0])
        XCTAssertEqual(run.description, "Panel: 3 columns, 3 rows")
    }

    // Full pipeline: split, then walk the test rows one record at a time
    func testSplitThenIterateTestRows() {
        let data = Panel([
            ("feature", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
            ("label", [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        ])
        let (_, test) = data.trainTestSplit(testRatio: 0.2, seed: 42)
        var features: [Double] = []
        for row in test {
            features.append(row["feature"])
        }
        XCTAssertEqual(features, test["feature"])
    }
}
