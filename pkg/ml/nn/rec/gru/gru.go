// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gru

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	WPart    nn.Param `type:"weights"`
	WPartRec nn.Param `type:"weights"`
	BPart    nn.Param `type:"biases"`
	WRes     nn.Param `type:"weights"`
	WResRec  nn.Param `type:"weights"`
	BRes     nn.Param `type:"biases"`
	WCand    nn.Param `type:"weights"`
	WCandRec nn.Param `type:"weights"`
	BCand    nn.Param `type:"biases"`
}

// New returns a new model with parameters initialized to zeros.
func New(in, out int) *Model {
	var m Model
	m.WPart, m.WPartRec, m.BPart = newGateParams(in, out)
	m.WRes, m.WResRec, m.BRes = newGateParams(in, out)
	m.WCand, m.WCandRec, m.BCand = newGateParams(in, out)
	return &m
}

func newGateParams(in, out int) (w, wRec, b nn.Param) {
	w = nn.NewParam(mat.NewEmptyDense(out, in))
	wRec = nn.NewParam(mat.NewEmptyDense(out, out))
	b = nn.NewParam(mat.NewEmptyVecDense(out))
	return
}

// State represent a state of the GRU recurrent network.
type State struct {
	R ag.Node
	P ag.Node
	C ag.Node
	Y ag.Node
}

// Processor implements the nn.Processor interface for a GRU Model.
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
		log.Fatal("gru: the initial state must be set before any input")
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

// LastState returns the last state of the recurrent network.
// It returns nil if there are no states.
func (p *Processor) LastState() *State {
	n := len(p.States)
	if n == 0 {
		return nil
	}
	return p.States[n-1]
}

// r = sigmoid(wr (dot) x + br + wrRec (dot) yPrev)
// p = sigmoid(wp (dot) x + bp + wpRec (dot) yPrev)
// c = f(wc (dot) x + bc + wcRec (dot) (yPrev * r))
// y = p * c + (1 - p) * yPrev
func (p *Processor) forward(x ag.Node) (s *State) {
	m := p.Model.(*Model)
	g := p.Graph
	s = new(State)
	yPrev := p.prev()
	s.R = g.Sigmoid(nn.Affine(g, m.BRes, m.WRes, x, m.WResRec, yPrev))
	s.P = g.Sigmoid(nn.Affine(g, m.BPart, m.WPart, x, m.WPartRec, yPrev))
	s.C = g.Tanh(nn.Affine(g, m.BCand, m.WCand, x, m.WCandRec, tryProd(g, yPrev, s.R)))
	s.Y = g.Prod(s.P, s.C)
	if yPrev != nil {
		s.Y = g.Add(s.Y, g.Prod(g.ReverseSub(s.P, g.NewScalar(1.0)), yPrev))
	}
	return
}

func (p *Processor) prev() (yPrev ag.Node) {
	s := p.LastState()
	if s != nil {
		yPrev = s.Y
	}
	return
}

// tryProd returns the product if 'a' il not nil, otherwise nil
func tryProd(g *ag.Graph, a, b ag.Node) ag.Node {
	if a != nil {
		return g.Prod(a, b)
	}
	return nil
}
