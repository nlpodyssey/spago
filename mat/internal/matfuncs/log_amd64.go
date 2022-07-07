// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && gc && !purego

package matfuncs

import "math"

// Log32 computes the natural logarithm of each element of x, storing the result in y (32 bits).
func Log32(x, y []float32) {
	if len(x) == 0 {
		return
	}

	if hasAVX2 {
		_ = y[len(x)-1]
		max := len(x) - 8
		for i := 0; i <= max; i += 8 {
			LogAVX32(x[i:], y[i:])
		}

		if max > 0 {
			for i, v := range x[:max+1] {
				if v == 0.0 {
					y[i] = float32(math.Inf(-1))
				}
			}
		}

		mod := len(x) % 8
		if mod > 0 {
			tailStart := len(x) - mod
			log(x[tailStart:], y[tailStart:])
		}
		return
	}

	_ = y[len(x)-1]
	max := len(x) - 4
	for i := 0; i <= max; i += 4 {
		LogSSE32(x[i:], y[i:])
	}

	if max > 0 {
		for i, v := range x[:max+1] {
			if v == 0.0 {
				y[i] = float32(math.Inf(-1))
			}
		}
	}

	mod := len(x) % 4
	if mod > 0 {
		tailStart := len(x) - mod
		log(x[tailStart:], y[tailStart:])
	}
}

// Log64 computes the natural logarithm of each element of x, storing the result in y (64 bits).
func Log64(x, y []float64) {
	log(x, y)
}
