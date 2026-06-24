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

import Foundation

// MARK: - Vector Form

/// The visual form a vector takes when rendered via `asExpression()`.
public enum VectorForm: Sendable {
    /// Column form with stacked Unicode brackets — the textbook default.
    case column
    /// Inline angle-bracket form, `⟨a, b, c⟩` — the prose-friendly variant.
    case inline
}
