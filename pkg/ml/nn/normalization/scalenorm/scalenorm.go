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

// Model contains the serializable parameters.
type Model struct {
	Gain *nn.Param `type:"weights"`
}

// New returns a new model with parameters initialized to zeros.
func New(size int) *Model {
	return &Model{
		Gain: nn.NewParam(mat.NewEmptyVecDense(size)),
	}
}

type Processor struct {
	nn.BaseProcessor
	gain ag.Node
	eps  ag.Node
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
		gain: ctx.Graph.NewWrap(m.Gain),
		eps:  ctx.Graph.Constant(1e-10),
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	g := p.Graph
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		norm := g.Sqrt(g.ReduceSum(g.Square(x)))
		ys[i] = g.Prod(g.DivScalar(x, g.AddScalar(norm, p.eps)), p.gain)
	}
	return ys
}
