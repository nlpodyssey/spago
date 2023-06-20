// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attention

import (
	"math"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// ScaledDotProductAttention is a self-attention mechanism relating different positions of a single
// sequence to compute a representation of the same sequence.
// This method requires that the query, the key and the value vectors have already been obtained
// from the input sequence. The scaled factor is the square root of the dimension of the key vectors.
func ScaledDotProductAttention(q []mat.Tensor, k, v, scaleFactor mat.Tensor, useCausalMask bool) ([]mat.Tensor, []mat.Tensor) {
	nodes := make([]mat.Tensor, len(q)*2)
	attention := nodes[:len(q)]
	weights := nodes[len(q):]

	causalMaskEnabled := useCausalMask && len(q) > 1
	kRows := k.Value().Shape()[0]

	kqi := make([]mat.Tensor, len(q))
	for i, qi := range q {
		kqi[i] = ag.Mul(k, qi)
	}

	for i, kqii := range kqi {
		scores := ag.ProdScalar(kqii, scaleFactor)

		if causalMaskEnabled {
			causalMask := k.Value().(mat.Matrix).NewMatrix(mat.WithBacking(makeCausalMask(i, kRows))) // TODO: use external cache for causal mask?
			scores = ag.Add(scores, causalMask)
		}

		weights[i] = ag.Softmax(scores)
	}

	for i, w := range weights {
		attention[i] = ag.MulT(v, w)
	}

	return attention, weights
}

// makeCausalMask returns a slice of size seqLength filled with zeros until curIndex, and the rest with -inf.
// FIXME: avoid specific float64 type, later passed to NewVec
func makeCausalMask(curIndex, seqLength int) []float64 {
	negInf := math.Inf(-1)
	causalMask := make([]float64, seqLength)
	for k := curIndex + 1; k < seqLength; k++ {
		causalMask[k] = negInf
	}
	return causalMask
}
