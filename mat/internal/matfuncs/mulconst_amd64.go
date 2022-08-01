// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && gc && !purego

package matfuncs

var (
	mulConst32 = MulConstSSE32
	mulConst64 = MulConstSSE64
)

func init() {
	if hasAVX2 {
		mulConst32 = MulConstAVX32
		mulConst64 = MulConstAVX64
	}
}

// MulConst32 multiplies each element of x by a constant value c, storing the result in y (32 bits).
func MulConst32(c float32, x, y []float32) {
	mulConst32(c, x, y)
}

// MulConst64 multiplies each element of x by a constant value c, storing the result in y (64 bits).
func MulConst64(c float64, x, y []float64) {
	mulConst64(c, x, y)
}
