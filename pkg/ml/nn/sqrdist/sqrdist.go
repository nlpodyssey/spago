// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sqrdist

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	B *nn.Param `type:"weights"`
}

func New(in, rank int) *Model {
	return &Model{
		B: nn.NewParam(mat.NewEmptyDense(rank, in)),
	}
}

type Processor struct {
	nn.BaseProcessor
	b ag.Node
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		b: g.NewWrap(m.B),
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.forward(x)
	}
	return ys
}

func (p *Processor) forward(x ag.Node) ag.Node {
	g := p.Graph
	bh := g.Mul(p.b, x)
	return g.Mul(g.T(bh), bh)
}
