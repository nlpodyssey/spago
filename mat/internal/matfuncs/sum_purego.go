// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || !gc || purego

package matfuncs

// Sum32 returns the sum of all values of x (32 bits).
func Sum32(x []float32) float32 {
	return sum(x)
}

// Sum64 returns the sum of all values of x (64 bits).
func Sum64(x []float64) float64 {
	return sum(x)
}

func sum[F float32 | float64](x []F) (y F) {
	for _, v := range x {
		y += v
	}
	return
}
