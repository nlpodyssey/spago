// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attention

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"sync"
)

// QKV groups queries, keys and values useful for self-attention functions, as described in "Attention Is
// All You Need" (Vaswani et al., 2017 - http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).
type QKV[T mat.DType] struct {
	Queries []ag.Node[T]
	Keys    []ag.Node[T]
	Values  []ag.Node[T]
}

// Output aggregates the multiple output of the self-attentions,
// incl. attention scores and last projected keys and values.
type Output[T mat.DType] struct {
	// AttOutput is the result of the self-attention.
	AttOutput []ag.Node[T]
	// AttWeights are the attention scores for each element of the sequence.
	AttWeights []mat.Matrix[T]
	// ProjKeysValues is the list of Keys and Values used to compute the self-attention.
	ProjKeysValues KeysValuesPair[T]
}

// KeysValuesPair contains Keys and Values.
type KeysValuesPair[T mat.DType] struct {
	Keys   []ag.Node[T]
	Values []ag.Node[T]
}

// ToQKV create a new QKV struct with queries = keys = values = xs.
func ToQKV[T mat.DType](xs []ag.Node[T]) QKV[T] {
	return QKV[T]{
		Queries: xs,
		Keys:    xs,
		Values:  xs,
	}
}

// ScaledDotProductAttention is a self-attention mechanism relating different positions of a single
// sequence to compute a representation of the same sequence.
// This method requires that the query, the key and the value vectors have already been obtained
// from the input sequence. The scaled factor is the square root of the dimension of the key vectors.
func ScaledDotProductAttention[T mat.DType](
	qkv QKV[T],
	scaleFactor T,
	useCausalMask bool,
) (context []ag.Node[T], prob []mat.Matrix[T]) {
	context = make([]ag.Node[T], len(qkv.Queries))
	prob = make([]mat.Matrix[T], len(qkv.Queries))
	keys := ag.Stack(qkv.Keys...)
	values := ag.T(ag.Stack(qkv.Values...))
	factor := keys.Graph().Constant(scaleFactor)

	for i, q := range qkv.Queries {
		attScores := ag.ProdScalar(ag.Mul(keys, q), factor)

		if useCausalMask && len(qkv.Queries) > 1 {
			causalMask := MakeCausalMask[T](i, len(qkv.Keys)) // TODO: use external cache for causal mask?
			attScores = ag.Add(attScores, attScores.Graph().NewVariable(mat.NewVecDense[T](causalMask), false))
		}

		attProb := ag.Softmax(attScores)
		context[i] = ag.Mul(values, attProb)
		prob[i] = attProb.Value()
	}
	return
}

// MakeCausalMask returns a slice of size seqLength filled with zeros until curIndex, and the rest with -inf.
func MakeCausalMask[T mat.DType](curIndex, seqLength int) []T {
	causalMask := make([]T, seqLength)
	for k := curIndex + 1; k < seqLength; k++ {
		causalMask[k] = mat.Inf[T](-1)
	}
	return causalMask
}

// ScaledDotProductAttentionConcurrent does the same thing as ScaledDotProductAttention but processes input concurrently.
func ScaledDotProductAttentionConcurrent[T mat.DType](qkv QKV[T], scaleFactor T) (context []ag.Node[T], prob []mat.Matrix[T]) {
	context = make([]ag.Node[T], len(qkv.Queries))
	prob = make([]mat.Matrix[T], len(qkv.Queries))
	keys := ag.Stack(qkv.Keys...)
	values := ag.T(ag.Stack(qkv.Values...))
	factor := keys.Graph().Constant(scaleFactor)
	var wg sync.WaitGroup
	wg.Add(len(qkv.Queries))
	for i, q := range qkv.Queries {
		go func(i int, q ag.Node[T]) {
			defer wg.Done()
			attScores := ag.ProdScalar(ag.Mul(keys, q), factor)
			attProb := ag.Softmax(attScores)
			context[i] = ag.Mul(values, attProb)
			prob[i] = attProb.Value()
		}(i, q)
	}
	wg.Wait()
	return
}

// MappingFunc is a mapping function used by LinearAttention.
type MappingFunc[T mat.DType] func(x ag.Node[T]) ag.Node[T]

// LinearAttention performs the self-attention as a linear dot-product of kernel feature maps.
// It operates with O(N) complexity, where N is the sequence length.
// Reference: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" by Katharopoulos et al. (2020)
func LinearAttention[T mat.DType](qkv QKV[T], mappingFunction MappingFunc[T], eps T) []ag.Node[T] {
	context := make([]ag.Node[T], len(qkv.Queries))
	attKeys := make([]ag.Node[T], len(qkv.Keys))

	var attKeysSum ag.Node[T] = nil
	for i := range qkv.Keys {
		attKeys[i] = mappingFunction(qkv.Keys[i])
		attKeysSum = ag.Add(attKeysSum, attKeys[i])
	}

	attKeysT := ag.T(ag.Stack(attKeys...))
	kv := ag.Mul(attKeysT, ag.Stack(qkv.Values...))

	epsn := kv.Graph().Constant(eps)
	for i := range qkv.Queries {
		attQuery := mappingFunction(qkv.Queries[i])
		n := ag.T(ag.Mul(ag.T(attQuery), kv))
		d := ag.Dot(attQuery, attKeysSum)
		context[i] = ag.DivScalar(n, ag.AddScalar(d, epsn))
	}
	return context
}
