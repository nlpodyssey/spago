// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package horn provides an implementation of Higher Order Recurrent Neural Networks (HORN).
package horn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
	"math"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	W    nn.Param   `type:"weights"`
	WRec []nn.Param `type:"weights"`
	B    nn.Param   `type:"biases"`
}

// New returns a new model with parameters initialized to zeros.
func New(in, out, order int) *Model {
	wRec := make([]nn.Param, order, order)
	for i := 0; i < order; i++ {
		wRec[i] = nn.NewParam(mat.NewEmptyDense(out, out))
	}
	return &Model{
		W:    nn.NewParam(mat.NewEmptyDense(out, in)),
		WRec: wRec,
		B:    nn.NewParam(mat.NewEmptyVecDense(out)),
	}
}

// State represent a state of the Horn recurrent network.
type State struct {
	Y ag.Node
}

// Processor implements the nn.Processor interface for a HORN Model.
type Processor struct {
	nn.BaseProcessor
	States []*State
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.NewBaseProcessor(m, ctx, false),
		States:        nil,
	}
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("horn: the initial state must be set before any input")
	}
	p.States = append(p.States, state)
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		s := p.forward(x)
		p.States = append(p.States, s)
		ys[i] = s.Y
	}
	return ys
}

func (p *Processor) forward(x ag.Node) (s *State) {
	m := p.Model.(*Model)
	s = new(State)
	h := nn.Affine(p.Graph, append([]ag.Node{m.B, m.W, x}, p.feedback()...)...)
	s.Y = p.Graph.Tanh(h)
	return
}

func (p *Processor) feedback() []ag.Node {
	m := p.Model.(*Model)
	var ys []ag.Node
	n := len(p.States)
	for i := 0; i < utils.MinInt(len(m.WRec), n); i++ {
		alpha := p.Graph.NewScalar(math.Pow(0.6, float64(i+1)))
		ys = append(ys, m.WRec[i], p.Graph.ProdScalar(p.States[n-1-i].Y, alpha))
	}
	return ys
}
