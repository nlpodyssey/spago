// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package highway

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
	WIn        nn.Param `type:"weights"`
	BIn        nn.Param `type:"biases"`
	WT         nn.Param `type:"weights"`
	BT         nn.Param `type:"biases"`
	Activation ag.OpName
}

// New returns a new model with parameters initialized to zeros.
func New(in int, activation ag.OpName) *Model {
	return &Model{
		WIn:        nn.NewParam(mat.NewEmptyDense(in, in)),
		BIn:        nn.NewParam(mat.NewEmptyVecDense(in)),
		WT:         nn.NewParam(mat.NewEmptyDense(in, in)),
		BT:         nn.NewParam(mat.NewEmptyVecDense(in)),
		Activation: activation,
	}
}

// Processor implements the nn.Processor interface for a highway Model.
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

// t = sigmoid(wT (dot) x + bT)
// h = f(wIn (dot) x + bIn)
// y = t * h + (1 - t) * x
func (p *Processor) forward(x ag.Node) ag.Node {
	m := p.Model.(*Model)
	activation := m.Activation
	g := p.Graph
	t := g.Sigmoid(nn.Affine(g, m.BT, m.WT, x))
	h := g.Invoke(activation, nn.Affine(g, m.BIn, m.WIn, x))
	y := g.Add(g.Prod(t, h), g.Prod(g.ReverseSub(t, g.NewScalar(1.0)), x))
	return y
}
