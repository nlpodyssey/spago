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
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config
	Query     *linear.Model
	R         nn.Param `spago:"type:weights"`
	Value     *linear.Model
	Attention *ContextProb `spago:"scope:processor"`
}

// Config provides configuration settings for a LSH-Attention Model.
type Config struct {
	InputSize   int
	QuerySize   int
	ValueSize   int
	BucketSize  int // num of buckets / 2
	ScaleFactor mat.Float
}

// ContextProb is a pair of Context encodings and Prob attention scores.
type ContextProb struct {
	// Context encodings.
	Context []ag.Node
	// Prob attention scores.
	Prob []mat.Matrix
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New(config Config) *Model {
	return &Model{
		Config: config,
		Query:  linear.New(config.InputSize, config.QuerySize),
		R:      nn.NewParam(mat.NewEmptyDense(config.QuerySize, config.BucketSize)),
		Value:  linear.New(config.InputSize, config.ValueSize),
	}
}

type indexedNodes struct {
	node  []ag.Node
	index []int
}

// getHash returns the hash for the dense matrix `x`.
// Since the hash does not require the use of gradients, it is calculated outside the graph to reduce overhead.
func (m *Model) getHash(x mat.Matrix) int {
	h := x.T().Mul(m.R.Value())
	concat := mat.ConcatV(h, h.ProdScalar(-1.0))
	return floatutils.ArgMax(concat.Data())
}

// TODO: implement concurrent computation?
func (m *Model) lshScaledDotProductAttention(
	g *ag.Graph,
	q ag.Node,
	ks,
	vs *indexedNodes,
	length int,
	scaleFactor mat.Float,
) (context ag.Node, prob mat.Matrix) {
	prob = mat.NewEmptyVecDense(length)
	keys := g.Stack(ks.node...)
	values := g.T(g.Stack(vs.node...))
	factor := g.NewScalar(scaleFactor)

	attScores := g.ProdScalar(g.Mul(keys, q), factor)
	attProb := g.Softmax(attScores)
	context = g.Mul(values, attProb)

	probData := prob.Data()
	attProbData := attProb.Value().Data()
	for j, i := range ks.index {
		probData[i] = attProbData[j]
	}
	return context, prob
}

func insertNode(m map[int]*indexedNodes, node ag.Node, i, h int) {
	if _, found := m[h]; !found {
		m[h] = &indexedNodes{node: []ag.Node{}, index: []int{}}
	}
	element := m[h]
	element.node = append(element.node, node)
	element.index = append(element.index, i)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()
	length := len(xs)
	qs := m.Query.Forward(xs...)
	ks := make([]ag.Node, length)
	vs := m.Value.Forward(xs...)
	mapk := make(map[int]*indexedNodes)
	mapv := make(map[int]*indexedNodes)

	// TODO: can it be implemented in a concurrent fashion?
	for i, q := range qs {
		norm := g.Sqrt(g.ReduceSum(g.Pow(q, 2.0)))
		ks[i] = g.DivScalar(q, norm) // Euclidean norm
		h := m.getHash(ks[i].Value())
		insertNode(mapk, ks[i], i, h)
		insertNode(mapv, vs[i], i, h)
	}

	context := make([]ag.Node, length)
	prob := make([]mat.Matrix, length)
	for i, q := range qs {
		j := m.getHash(q.Value())
		c, p := m.lshScaledDotProductAttention(g, q, mapk[j], mapv[j], length, m.Config.ScaleFactor)
		context[i], prob[i] = c, p
	}

	m.Attention = &ContextProb{
		Context: context,
		Prob:    prob,
	}
	return context
}
