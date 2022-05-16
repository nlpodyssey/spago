// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package float

// DType is the primary type constraint for matrices defined in this package.
type DType interface {
	float32 | float64
}
