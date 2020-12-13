// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fofe

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/utils"
)

func EncodeDense(alpha float64, size int, seq []int) []*mat.Dense {
	y := make([]*mat.Dense, len(seq), len(seq))
	for i, x := range Encode(alpha, size, seq) {
		y[i] = mat.NewVecDense(x.Data())
	}
	return y
}

// Encode implements the Fixed-Size Ordinally-Forgetting Encoding.
// zt = α * zt−1 + et (1 ≤ t ≤ T)
func Encode(alpha float64, size int, seq []int) []*mat.Sparse {
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

func BiEncode(alpha float64, size int, seq []int) (fwd []*mat.Sparse, bwd []*mat.Sparse) {
	fwd = Encode(alpha, size, seq)
	bwd = Encode(alpha, size, utils.ReverseIntSlice(seq))
	utils.ReverseInPlace(bwd)
	return
}
