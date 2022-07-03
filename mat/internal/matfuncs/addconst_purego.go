// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || !gc || purego

package matfuncs

// AddConst32 adds a constant value c to each element of x, storing the result in y (32 bits).
func AddConst32(c float32, x, y []float32) {
	addConst(c, x, y)
}

// AddConst64 adds a constant value c to each element of x, storing the result in y (64 bits).
func AddConst64(c float64, x, y []float64) {
	addConst(c, x, y)
}

func addConst[F float32 | float64](c F, x, y []F) {
	if len(x) == 0 {
		return
	}
	_ = y[len(x)-1]
	for i, xv := range x {
		y[i] = xv + c
	}
}
