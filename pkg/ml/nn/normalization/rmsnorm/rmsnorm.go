// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reference: "Root Mean Square Layer Normalization" by Biao Zhang and Rico Sennrich (2019).
// (https://arxiv.org/pdf/1910.07467.pdf)
package rmsnorm

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	W *nn.Param `type:"weights"`
	B *nn.Param `type:"biases"`
}

// New returns a new model with parameters initialized to zeros.
func New(size int) *Model {
	return &Model{
		W: nn.NewParam(mat.NewEmptyVecDense(size)),
		B: nn.NewParam(mat.NewEmptyVecDense(size)),
	}
}

type Processor struct {
	nn.BaseProcessor
	w   ag.Node
	b   ag.Node
	eps ag.Node
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		w:   ctx.Graph.NewWrap(m.W),
		b:   ctx.Graph.NewWrap(m.B),
		eps: ctx.Graph.Constant(1e-10),
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	g := p.Graph
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		rms := g.Sqrt(g.ReduceMean(g.Square(x)))
		ys[i] = g.Add(g.Prod(g.DivScalar(x, g.AddScalar(rms, p.eps)), p.w), p.b)
	}
	return ys
}
