// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scalenorm

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
	Gain *nn.Param `type:"weights"`
}

func New(size int) *Model {
	return &Model{
		Gain: nn.NewParam(mat.NewEmptyVecDense(size)),
	}
}

type Processor struct {
	nn.BaseProcessor
	gain ag.Node
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		gain: g.NewWrap(m.Gain),
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	g := p.Graph
	ys := make([]ag.Node, len(xs))
	eps := g.NewScalar(1e-10)
	for i, x := range xs {
		norm := g.Sqrt(g.ReduceSum(g.Square(x)))
		ys[i] = g.Prod(g.DivScalar(x, g.AddScalar(norm, eps)), p.gain)
	}
	return ys
}
