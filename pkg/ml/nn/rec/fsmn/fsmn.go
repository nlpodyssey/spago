// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fsmn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model implements a variant of the Feedforward Sequential Memory Networks
// (https://arxiv.org/pdf/1512.08301.pdf) where the neurons in the same hidden layer
// are independent of each other and they are connected across layers as in the IndRNN.
type Model struct {
	W     nn.Param   `type:"weights"`
	WRec  nn.Param   `type:"weights"`
	WS    []nn.Param `type:"weights"` // coefficient vectors for scaling
	B     nn.Param   `type:"biases"`
	order int
}

// New returns a new model with parameters initialized to zeros.
func New(in, out, order int) *Model {
	WS := make([]nn.Param, order, order)
	for i := 0; i < order; i++ {
		WS[i] = nn.NewParam(mat.NewEmptyVecDense(out))
	}
	return &Model{
		W:     nn.NewParam(mat.NewEmptyDense(out, in)),
		WRec:  nn.NewParam(mat.NewEmptyVecDense(out)),
		WS:    WS,
		B:     nn.NewParam(mat.NewEmptyVecDense(out)),
		order: order,
	}
}

// State represent a state of the FSMN recurrent network.
type State struct {
	Y ag.Node
}

// Processor implements the nn.Processor interface for an FSMN Model.
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
		log.Fatal("fsmn: the initial state must be set before any input")
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
	g := p.Graph
	s = new(State)
	h := nn.Affine(g, m.B, m.W, x)
	if len(p.States) > 0 {
		h = g.Add(h, g.Prod(m.WRec, p.feedback()))
	}
	s.Y = g.ReLU(h)
	return
}

func (p *Processor) feedback() ag.Node {
	m := p.Model.(*Model)
	g := p.Graph
	var y ag.Node
	n := len(p.States)
	min := utils.MinInt(m.order, n)
	for i := 0; i < min; i++ {
		scaled := g.Prod(m.WS[i], g.NewWrapNoGrad(p.States[n-1-i].Y))
		if y == nil {
			y = scaled
		} else {
			y = g.Add(y, scaled)
		}
	}
	return y
}
