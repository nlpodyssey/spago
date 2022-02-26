// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attention

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// ScaledDotProductAttention is a self-attention mechanism relating different positions of a single
// sequence to compute a representation of the same sequence.
// This method requires that the query, the key and the value vectors have already been obtained
// from the input sequence. The scaled factor is the square root of the dimension of the key vectors.
func ScaledDotProductAttention[T mat.DType](q, k, v []ag.Node[T], scaleFactor T, useCausalMask bool) ([]ag.Node[T], []ag.Node[T]) {
	attention := make([]ag.Node[T], len(q))
	weights := make([]ag.Node[T], len(q))
	keys := ag.Stack(k...)
	values := ag.T(ag.Stack(v...))
	factor := keys.Graph().Constant(scaleFactor)

	for i, qi := range q {
		scores := ag.ProdScalar(ag.Mul(keys, qi), factor)

		if useCausalMask && len(q) > 1 {
			causalMask := mat.NewVecDense[T](makeCausalMask[T](i, len(k))) // TODO: use external cache for causal mask?
			scores = ag.Add(scores, scores.Graph().NewVariable(causalMask, false))
		}

		weights[i] = ag.Softmax(scores)
		attention[i] = ag.Mul(values, weights[i])
	}

	return attention, weights
}

// makeCausalMask returns a slice of size seqLength filled with zeros until curIndex, and the rest with -inf.
func makeCausalMask[T mat.DType](curIndex, seqLength int) []T {
	causalMask := make([]T, seqLength)
	for k := curIndex + 1; k < seqLength; k++ {
		causalMask[k] = mat.Inf[T](-1)
	}
	return causalMask
}

// MappingFunc is a mapping function used by LinearAttention.
type MappingFunc[T mat.DType] func(x ag.Node[T]) ag.Node[T]

// LinearAttention performs the self-attention as a linear dot-product of kernel feature maps.
// It operates with O(N) complexity, where N is the sequence length.
// Reference: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" by Katharopoulos et al. (2020)
func LinearAttention[T mat.DType](q, k, v []ag.Node[T], mappingFunction MappingFunc[T], eps T) []ag.Node[T] {
	context := make([]ag.Node[T], len(q))
	attKeys := make([]ag.Node[T], len(k))

	var attKeysSum ag.Node[T] = nil
	for i := range k {
		attKeys[i] = mappingFunction(k[i])
		attKeysSum = ag.Add(attKeysSum, attKeys[i])
	}

	attKeysT := ag.T(ag.Stack(attKeys...))
	kv := ag.Mul(attKeysT, ag.Stack(v...))

	epsn := kv.Graph().Constant(eps)
	for i, qi := range q {
		attQuery := mappingFunction(qi)
		n := ag.T(ag.Mul(ag.T(attQuery), kv))
		d := ag.Dot(attQuery, attKeysSum)
		context[i] = ag.DivScalar(n, ag.AddScalar(d, epsn))
	}
	return context
}
