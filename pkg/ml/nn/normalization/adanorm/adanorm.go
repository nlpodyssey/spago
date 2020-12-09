// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reference: "Understanding and Improving Layer Normalization" by Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, Junyang Lin (2019).
// (https://papers.nips.cc/paper/8689-understanding-and-improving-layer-normalization.pdf)
package adanorm

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the scaling factor.
type Model struct {
	scale float64
}

// New returns a new model.
func New(scale float64) *Model {
	return &Model{scale: scale}
}

type Processor struct {
	nn.BaseProcessor
	eps ag.Node
	one ag.Node
	k   ag.Node
	c   ag.Node
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	g := ctx.Graph
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		eps: g.Constant(1e-10),
		one: g.Constant(1.0),
		k:   g.Constant(0.1),
		c:   g.Constant(m.scale),
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	g := p.Graph
	meanVectors := p.Mean(xs)
	devVectors := p.StdDev(meanVectors, xs)
	zs := make([]ag.Node, len(xs))

	for i, x := range xs {
		y := g.DivScalar(g.SubScalar(x, meanVectors[i]), g.Add(devVectors[i], p.eps))
		fi := g.ProdScalar(g.ReverseSub(g.ProdScalar(y, p.k), p.one), p.c)
		zs[i] = g.Prod(y, g.NewWrapNoGrad(fi)) // detach the gradient of fi and only treat it as a changeable constant in implementation
	}
	return zs
}

func (p *Processor) Mean(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.Graph.ReduceMean(x)
	}
	return ys
}

func (p *Processor) StdDev(meanVectors []ag.Node, xs []ag.Node) []ag.Node {
	g := p.Graph
	devVectors := make([]ag.Node, len(xs))
	for i, x := range xs {
		diffVector := g.Square(g.SubScalar(x, meanVectors[i]))
		devVectors[i] = g.Sqrt(g.ReduceMean(diffVector))
	}
	return devVectors
}
