// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import "math"

// Float is the main float type for the mat package. It is an alias for float64.
type Float = float64

// SmallestNonzeroFloat corresponds to math.SmallestNonzeroFloat64.
const SmallestNonzeroFloat Float = math.SmallestNonzeroFloat64
