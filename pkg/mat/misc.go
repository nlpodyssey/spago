// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"math"
)

func SameDims(a, b Matrix) bool {
	r, c := a.Dims()
	r2, c2 := b.Dims()
	return r == r2 && c == c2
}

func SameSize(a, b Matrix) bool {
	return a.Size() == b.Size()
}

func VectorsOfSameSize(a, b Matrix) bool {
	return SameSize(a, b) && a.IsVector() && b.IsVector()
}

func Sqrt(m Matrix) Matrix {
	buf := m.ZerosLike()
	buf.Apply(func(i, j int, v float64) float64 {
		return math.Sqrt(v)
	}, m)
	return buf
}

func Print(a Matrix) {
	r, c := a.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			fmt.Printf("%.8f ", a.At(i, j))
		}
	}
	fmt.Printf("\n")
}

// Cosine returns the cosine similarity between two vectors.
func Cosine(x, y Matrix) float64 {
	d := x.DotUnitary(y)
	xNorm := x.Norm(2.0)
	yNorm := y.Norm(2.0)
	return d / (xNorm * yNorm)
}
