// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attention

import (
	"math"
	"sync"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
)

// ScaledDotProductAttention is a self-attention mechanism relating different positions of a single
// sequence to compute a representation of the same sequence.
// This method requires that the query, the key and the value vectors have already been obtained
// from the input sequence. The scaled factor is the square root of the dimension of the key vectors.
func ScaledDotProductAttention(q []ag.Node, k, v ag.Node, scaleFactor float64, useCausalMask bool) ([]ag.Node, []ag.Node) {
	attention := make([]ag.Node, len(q))
	weights := make([]ag.Node, len(q))
	factor := ag.Var(k.Value().NewScalar(float.Interface(scaleFactor)))

	causalMaskEnabled := useCausalMask && len(q) > 1
	kRows := k.Value().Rows()

	var wg sync.WaitGroup
	wg.Add(len(q))

	for i, qi := range q {
		go func(i int, qi ag.Node) {
			scores := ag.ProdScalar(ag.Mul(k, qi), factor)

			if causalMaskEnabled {
				causalMask := k.Value().NewVec(float.SliceInterface(makeCausalMask(i, kRows))) // TODO: use external cache for causal mask?
				scores = ag.Add(scores, ag.Var(causalMask))
			}

			weights[i] = ag.Softmax(scores)
			attention[i] = ag.MulT(v, weights[i])
			wg.Done()
		}(i, qi)
	}

	wg.Wait()
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

// MappingFunc is a mapping function used by LinearAttention.
type MappingFunc func(x ag.Node) ag.Node

// LinearAttention performs the self-attention as a linear dot-product of kernel feature maps.
// It operates with O(N) complexity, where N is the sequence length.
// Reference: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" by Katharopoulos et al. (2020)
func LinearAttention(q, k, v []ag.Node, mappingFunction MappingFunc, eps float64) []ag.Node {
	if len(q) == 0 {
		return nil
	}
	context := make([]ag.Node, len(q))
	attKeys := make([]ag.Node, len(k))

	var attKeysSum ag.Node = nil
	for i := range k {
		attKeys[i] = mappingFunction(k[i])
		attKeysSum = ag.Add(attKeysSum, attKeys[i])
	}

	attKeysT := ag.T(ag.Stack(attKeys...))
	kv := ag.Mul(attKeysT, ag.Stack(v...))

	epsn := ag.Var(q[0].Value().NewScalar(float.Interface(eps)))
	for i, qi := range q {
		attQuery := mappingFunction(qi)
		n := ag.T(ag.Mul(ag.T(attQuery), kv))
		d := ag.Dot(attQuery, attKeysSum)
		context[i] = ag.DivScalar(n, ag.AddScalar(d, epsn))
	}
	return context
}
