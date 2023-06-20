// Copyright ©2016-2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f32

import "github.com/nlpodyssey/spago/mat/internal/f32/asm32"

// MatrixMul computes the matrix-matrix multiplication C = A * B.
// This code is adapted from Gonum's Dgemm implementation.
func MatrixMul(aRows, aCols, bCols int, a []float32, b []float32, c []float32) {
	for i := 0; i < aRows; i++ {
		ctmp := c[i*bCols : i*bCols+bCols]
		for l, v := range a[i*aCols : i*aCols+aCols] {
			if v == 0 {
				continue
			}
			asm32.AxpyUnitary(v, b[l*bCols:l*bCols+bCols], ctmp)
		}
	}
}

// AddConst is
//
//	for i := range x {
//		x[i] += alpha
//	}
func AddConst(alpha float32, x []float32) {
	for i := range x {
		x[i] += alpha
	}
}

// DivTo is
//
//	for i, v := range s {
//		dst[i] = v / t[i]
//	}
//	return dst
func DivTo(dst, s, t []float32) []float32 {
	for i, v := range s {
		dst[i] = v / t[i]
	}
	return dst
}

// CumSum is
//
//	if len(s) == 0 {
//		return dst
//	}
//	dst[0] = s[0]
//	for i, v := range s[1:] {
//		dst[i+1] = dst[i] + v
//	}
//	return dst
func CumSum(dst, s []float32) []float32 {
	if len(s) == 0 {
		return dst
	}
	dst[0] = s[0]
	for i, v := range s[1:] {
		dst[i+1] = dst[i] + v
	}
	return dst
}
