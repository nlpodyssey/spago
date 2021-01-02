// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fofe

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/utils"
)

// EncodeDense is similar to Encode, but it works with Dense matrices.
func EncodeDense(alpha mat.Float, size int, seq []int) []*mat.Dense {
	y := make([]*mat.Dense, len(seq), len(seq))
	for i, x := range Encode(alpha, size, seq) {
		y[i] = mat.NewVecDense(x.Data())
	}
	return y
}

// Encode is the FOFE encoding function, which works with Sparse matrices.
//
// Reference recursive formula: z(t) = α · z(t−1) + e(t), where 1 ≤ t ≤ T
func Encode(alpha mat.Float, size int, seq []int) []*mat.Sparse {
	var z []*mat.Sparse
	for t, i := range seq {
		x := mat.OneHotSparse(size, i)
		if len(z) > 0 {
			z = append(z, z[t-1].ProdScalar(alpha).Add(x).(*mat.Sparse))
		} else {
			z = append(z, x)
		}
	}
	return z
}

// BiEncode is the FOFE bidirectional encoding function.
func BiEncode(alpha mat.Float, size int, seq []int) (fwd []*mat.Sparse, bwd []*mat.Sparse) {
	fwd = Encode(alpha, size, seq)
	bwd = Encode(alpha, size, utils.ReverseIntSlice(seq))
	utils.ReverseInPlace(bwd)
	return
}
