// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && gc && !purego

package matfuncs

// DotProd32 returns the dot product between x1 and x2 (32 bits).
func DotProd32(x1, x2 []float32) float32 {
	if hasAVX {
		return DotProdAVX32(x1, x2)
	}
	return DotProdSSE32(x1, x2)
}

// DotProd64 returns the dot product between x1 and x2 (64 bits).
func DotProd64(x1, x2 []float64) float64 {
	if hasAVX {
		return DotProdAVX64(x1, x2)
	}
	return DotProdSSE64(x1, x2)
}
