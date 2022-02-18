// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lshattention provides an implementation of the LSH-Attention model, as
// describe in `Reformer: The Efficient Transformer` by N. Kitaev, Ł. Kaiser, A. Levskaya
// (https://arxiv.org/pdf/2001.04451.pdf).
// TODO: Check compatibility with the LSH Attention implemented by Hugging Face:
// TODO: https://huggingface.co/transformers/model_doc/reformer.html
package lshattention

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel
	Config[T]
	Query     *linear.Model[T]
	R         nn.Param[T] `spago:"type:weights"`
	Value     *linear.Model[T]
	Attention *ContextProb[T] `spago:"scope:processor"`
}

// Config provides configuration settings for a LSH-Attention Model.
type Config[T mat.DType] struct {
	InputSize   int
	QuerySize   int
	ValueSize   int
	BucketSize  int // num of buckets / 2
	ScaleFactor T
}

// ContextProb is a pair of Context encodings and Prob attention scores.
type ContextProb[T mat.DType] struct {
	// Context encodings.
	Context []ag.Node[T]
	// Prob attention scores.
	Prob []mat.Matrix[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](config Config[T]) *Model[T] {
	return &Model[T]{
		Config: config,
		Query:  linear.New[T](config.InputSize, config.QuerySize),
		R:      nn.NewParam[T](mat.NewEmptyDense[T](config.QuerySize, config.BucketSize)),
		Value:  linear.New[T](config.InputSize, config.ValueSize),
	}
}

type indexedNodes[T mat.DType] struct {
	node  []ag.Node[T]
	index []int
}

// getHash returns the hash for the dense matrix `x`.
// Since the hash does not require the use of gradients, it is calculated outside the graph to reduce overhead.
func (m *Model[T]) getHash(x mat.Matrix[T]) int {
	h := x.T().Mul(m.R.Value())
	concat := mat.ConcatV(h, h.ProdScalar(-1.0))
	return concat.VecArgMax()
}

// TODO: implement concurrent computation?
func (m *Model[T]) lshScaledDotProductAttention(
	q ag.Node[T],
	ks,
	vs *indexedNodes[T],
	length int,
	scaleFactor T,
) (context ag.Node[T], prob mat.Matrix[T]) {
	prob = mat.NewEmptyVecDense[T](length)
	keys := ag.Stack(ks.node...)
	values := ag.T(ag.Stack(vs.node...))
	factor := keys.Graph().NewScalar(scaleFactor)

	attScores := ag.ProdScalar(ag.Mul(keys, q), factor)
	attProb := ag.Softmax(attScores)
	context = ag.Mul[T](values, attProb)

	probData := prob.Data()
	attProbData := attProb.Value().Data()
	for j, i := range ks.index {
		probData[i] = attProbData[j]
	}
	return context, prob
}

func insertNode[T mat.DType](m map[int]*indexedNodes[T], node ag.Node[T], i, h int) {
	if _, found := m[h]; !found {
		m[h] = &indexedNodes[T]{node: []ag.Node[T]{}, index: []int{}}
	}
	element := m[h]
	element.node = append(element.node, node)
	element.index = append(element.index, i)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	length := len(xs)
	qs := m.Query.Forward(xs...)
	ks := make([]ag.Node[T], length)
	vs := m.Value.Forward(xs...)
	mapk := make(map[int]*indexedNodes[T])
	mapv := make(map[int]*indexedNodes[T])

	// TODO: can it be implemented in a concurrent fashion?
	for i, q := range qs {
		norm := ag.Sqrt(ag.ReduceSum(ag.Pow(q, 2.0)))
		ks[i] = ag.DivScalar(q, norm) // Euclidean norm
		h := m.getHash(ks[i].Value())
		insertNode(mapk, ks[i], i, h)
		insertNode(mapv, vs[i], i, h)
	}

	context := make([]ag.Node[T], length)
	prob := make([]mat.Matrix[T], length)
	for i, q := range qs {
		j := m.getHash(q.Value())
		c, p := m.lshScaledDotProductAttention(q, mapk[j], mapv[j], length, m.Config.ScaleFactor)
		context[i], prob[i] = c, p
	}

	m.Attention = &ContextProb[T]{
		Context: context,
		Prob:    prob,
	}
	return context
}
