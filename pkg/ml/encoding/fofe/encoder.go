// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fofe

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/utils"
)

// EncodeDense is similar to Encode, but it works with Dense matrices.
func EncodeDense[T mat.DType](alpha T, size int, seq []int) []mat.Matrix[T] {
	y := make([]mat.Matrix[T], len(seq), len(seq))
	for i, x := range Encode(alpha, size, seq) {
		y[i] = mat.NewVecDense(x.Data())
	}
	return y
}

// Encode is the FOFE encoding function, which works with Sparse matrices.
//
// Reference recursive formula: z(t) = α · z(t−1) + e(t), where 1 ≤ t ≤ T
func Encode[T mat.DType](alpha T, size int, seq []int) []mat.Matrix[T] {
	var z []mat.Matrix[T]
	for t, i := range seq {
		x := mat.NewOneHotVecDense[T](size, i) // FIXME: this was a sparse matrix!
		if len(z) > 0 {
			z = append(z, z[t-1].ProdScalar(alpha).Add(x))
		} else {
			z = append(z, x)
		}
	}
	return z
}

// BiEncode is the FOFE bidirectional encoding function.
func BiEncode[T mat.DType](alpha T, size int, seq []int) (fwd, bwd []mat.Matrix[T]) {
	fwd = Encode(alpha, size, seq)
	bwd = Encode(alpha, size, utils.ReverseIntSlice(seq))
	utils.ReverseInPlace(bwd)
	return
}
