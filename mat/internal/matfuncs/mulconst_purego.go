// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || !gc || purego

package matfuncs

// MulConst32 multiplies each element of x by a constant value c, storing the result in y (32 bits).
func MulConst32(c float32, x, y []float32) {
	mulConst(c, x, y)
}

// MulConst64 multiplies each element of x by a constant value c, storing the result in y (64 bits).
func MulConst64(c float64, x, y []float64) {
	mulConst(c, x, y)
}

func mulConst[F float32 | float64](c F, x, y []F) {
	if len(x) == 0 {
		return
	}
	_ = y[len(x)-1]
	for i, xv := range x {
		y[i] = xv * c
	}
}
