// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && gc && !purego

package matfuncs

// Exp32 computes the base-e exponential of each element of x, storing the result in y (32 bits).
func Exp32(x, y []float32) {
	if len(x) == 0 {
		return
	}

	if hasAVX2 {
		_ = y[len(x)-1]
		max := len(x) - 8
		for i := 0; i <= max; i += 8 {
			ExpAVX32(x[i:], y[i:])
		}

		mod := len(x) % 8
		if mod > 0 {
			tailStart := len(x) - mod
			exp(x[tailStart:], y[tailStart:])
		}
		return
	}

	_ = y[len(x)-1]
	max := len(x) - 4
	for i := 0; i <= max; i += 4 {
		ExpSSE32(x[i:], y[i:])
	}

	mod := len(x) % 4
	if mod > 0 {
		tailStart := len(x) - mod
		exp(x[tailStart:], y[tailStart:])
	}
}

// Exp64 computes the base-e exponential of each element of x, storing the result in y (64 bits).
func Exp64(x, y []float64) {
	exp(x, y)
}
