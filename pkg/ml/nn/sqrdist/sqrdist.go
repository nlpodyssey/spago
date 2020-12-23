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

// Model contains the serializable parameters.
type Model struct {
	B nn.Param `type:"weights"`
}

// New returns a new model with parameters initialized to zeros.
func New(in, rank int) *Model {
	return &Model{
		B: nn.NewParam(mat.NewEmptyDense(rank, in)),
	}
}

// Processor implements the nn.Processor interface for an sqrdist Model.
type Processor struct {
	nn.BaseProcessor
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.NewBaseProcessor(m, ctx, false),
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.forward(x)
	}
	return ys
}

func (p *Processor) forward(x ag.Node) ag.Node {
	m := p.Model.(*Model)
	g := p.Graph
	bh := g.Mul(m.B, x)
	return g.Mul(g.T(bh), bh)
}
