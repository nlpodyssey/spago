// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && gc && !purego

package matfuncs

var (
	sum32 = SumSSE32
	sum64 = SumSSE64
)

func init() {
	if hasAVX {
		sum32 = SumAVX32
		sum64 = SumAVX64
	}
}

// Sum32 returns the sum of all values of x (32 bits).
func Sum32(x []float32) float32 {
	return sum32(x)
}

// Sum64 returns the sum of all values of x (64 bits).
func Sum64(x []float64) float64 {
	return sum64(x)
}
