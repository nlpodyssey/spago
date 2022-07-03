// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && gc && !purego

package matfuncs

var (
	sub32 = SubSSE32
	sub64 = SubSSE64
)

func init() {
	if hasAVX {
		sub32 = SubAVX32
		sub64 = SubAVX64
	}
}

// Sub32 subtracts x2 from x1, element-wise, storing the result in y (32 bits).
func Sub32(x1, x2, y []float32) {
	sub32(x1, x2, y)
}

// Sub64 subtracts x2 from x1, element-wise, storing the result in y (64 bits).
func Sub64(x1, x2, y []float64) {
	sub64(x1, x2, y)
}
