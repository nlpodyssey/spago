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
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	Config
	Query *linear.Model
	R     *nn.Param `type:"weights"`
	Value *linear.Model
}

type Config struct {
	InputSize   int
	QuerySize   int
	ValueSize   int
	BucketSize  int // num of buckets / 2
	ScaleFactor float64
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

type ContextProb struct {
	context []ag.Node
	prob    []mat.Matrix
}

type Processor struct {
	nn.BaseProcessor
	scaleFactor float64
	query       *linear.Processor
	value       *linear.Processor
	r           ag.Node
	Attention   *ContextProb
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		scaleFactor: m.ScaleFactor,
		query:       m.Query.NewProc(ctx).(*linear.Processor),
		value:       m.Value.NewProc(ctx).(*linear.Processor),
		r:           ctx.Graph.NewWrap(m.R),
		Attention:   nil,
	}
}

type IndexedNodes struct {
	node  []ag.Node
	index []int
}

// getHash returns the hash for the dense matrix `x`.
// Since the hash does not require the use of gradients, it is calculated outside the graph to reduce overhead.
func (p *Processor) getHash(x *mat.Dense) int {
	h := x.T().Mul(p.r.Value())
	concat := mat.ConcatV(h, h.ProdScalar(-1.0))
	return f64utils.ArgMax(concat.Data())
}

// TODO: implement concurrent computation?
func (p *Processor) lshScaledDotProductAttention(
	g *ag.Graph,
	q ag.Node,
	ks,
	vs *IndexedNodes,
	length int,
	scaleFactor float64,
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

func insertNode(m map[int]*IndexedNodes, node ag.Node, i, h int) {
	if _, found := m[h]; !found {
		m[h] = &IndexedNodes{node: []ag.Node{}, index: []int{}}
	}
	element := m[h]
	element.node = append(element.node, node)
	element.index = append(element.index, i)
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	length := len(xs)
	qs := p.query.Forward(xs...)
	ks := make([]ag.Node, length)
	vs := p.value.Forward(xs...)
	mapk := make(map[int]*IndexedNodes)
	mapv := make(map[int]*IndexedNodes)

	// TODO: can it be implemented in a concurrent fashion?
	for i, q := range qs {
		norm := p.Graph.Sqrt(p.Graph.ReduceSum(p.Graph.Pow(q, 2.0)))
		ks[i] = p.Graph.DivScalar(q, norm) // Euclidean norm
		h := p.getHash(ks[i].Value().(*mat.Dense))
		insertNode(mapk, ks[i], i, h)
		insertNode(mapv, vs[i], i, h)
	}

	context := make([]ag.Node, length)
	prob := make([]mat.Matrix, length)
	for i, q := range qs {
		j := p.getHash(q.Value().(*mat.Dense))
		c, p := p.lshScaledDotProductAttention(p.Graph, q, mapk[j], mapv[j], length, p.scaleFactor)
		context[i], prob[i] = c, p
	}

	p.Attention = &ContextProb{
		context: context,
		prob:    prob,
	}
	return context
}
