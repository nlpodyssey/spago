// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || !gc || purego

package matfuncs

// Div32 divides x1 by x2, element-wise, storing the result in y (32 bits).
func Div32(x1, x2, y []float32) {
	div(x1, x2, y)
}

// Div64 divides x1 by x2, element-wise, storing the result in y (64 bits).
func Div64(x1, x2, y []float64) {
	div(x1, x2, y)
}

func div[F float32 | float64](x1, x2, y []F) {
	if len(x1) == 0 {
		return
	}
	_ = y[len(x1)-1]
	_ = x2[len(x1)-1]
	for i, x1v := range x1 {
		y[i] = x1v / x2[i]
	}
}
