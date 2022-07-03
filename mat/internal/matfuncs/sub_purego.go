// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || !gc || purego

package matfuncs

// Sub32 subtracts x2 from x1, element-wise, storing the result in y (32 bits).
func Sub32(x1, x2, y []float32) {
	sub(x1, x2, y)
}

// Sub64 subtracts x2 from x1, element-wise, storing the result in y (64 bits).
func Sub64(x1, x2, y []float64) {
	sub(x1, x2, y)
}

func sub[F float32 | float64](x1, x2, y []F) {
	if len(x1) == 0 {
		return
	}
	_ = y[len(x1)-1]
	_ = x2[len(x1)-1]
	for i, x1v := range x1 {
		y[i] = x1v - x2[i]
	}
}
