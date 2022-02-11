// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

// Float is an alias to one of the types constrained by DType.
//
// Deprecated: this type has been introduced as a temporary default type
// to be used during the development phase for the adoption of generics.
// It has the sole purpose of avoiding the adoption of explicit types (float32
// or float64) from all those occurrences in the code that are assuming
// a whole global type for all matrices and their values.
// Once the transition to generics will be complete, this type will be probably
// removed, or refactored into something else. You should not use this type
// unless you really know what you are doing!
type Float = float64
