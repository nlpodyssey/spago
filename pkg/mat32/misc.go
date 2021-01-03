// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"fmt"
	"math"
)

// SameDims returns whether the two matrices have the same number of rows and columns (so also of the same size).
func SameDims(a, b Matrix) bool {
	r, c := a.Dims()
	r2, c2 := b.Dims()
	return r == r2 && c == c2
}

// SameSize returns whether the two matrices have the same size (number of elements).
func SameSize(a, b Matrix) bool {
	return a.Size() == b.Size()
}

// VectorsOfSameSize returns whether the two matrices are vector of the same size.
func VectorsOfSameSize(a, b Matrix) bool {
	return SameSize(a, b) && a.IsVector() && b.IsVector()
}

// SqrtMatrix returns a new matrix filled with the sqrt of the values of the input matrix.
func SqrtMatrix(m Matrix) Matrix {
	buf := m.ZerosLike()
	buf.Apply(func(i, j int, v Float) Float {
		return Float(math.Sqrt(float64(v)))
	}, m)
	return buf
}

// Print performs a simple print of the matrix.
func Print(a Matrix) {
	r, c := a.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			fmt.Printf("%.8f ", a.At(i, j))
		}
	}
	fmt.Printf("\n")
}

// Cosine returns the cosine similarity between two not normalized vectors.
func Cosine(x, y Matrix) Float {
	d := x.DotUnitary(y)
	xNorm := x.Norm(2.0)
	yNorm := y.Norm(2.0)
	return d / (xNorm * yNorm)
}
