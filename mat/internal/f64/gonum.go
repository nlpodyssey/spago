// Copyright ©2016-2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f64

import "github.com/nlpodyssey/spago/mat/internal/f64/asm64"

// MatrixMul computes the matrix-matrix multiplication C = A * B.
// This code is adapted from Gonum's Dgemm implementation.
func MatrixMul(aRows, aCols, bCols int, a []float64, b []float64, c []float64) {
	for i := 0; i < aRows; i++ {
		ctmp := c[i*bCols : i*bCols+bCols]
		for l, v := range a[i*aCols : i*aCols+aCols] {
			if v == 0 {
				continue
			}
			asm64.AxpyUnitary(v, b[l*bCols:l*bCols+bCols], ctmp)
		}
	}
}
