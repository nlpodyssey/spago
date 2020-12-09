// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ltm

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
	W1    *nn.Param `type:"weights"`
	W2    *nn.Param `type:"weights"`
	W3    *nn.Param `type:"weights"`
	WCell *nn.Param `type:"weights"`
}

// New returns a new model with parameters initialized to zeros.
func New(in int) *Model {
	return &Model{
		W1:    nn.NewParam(mat.NewEmptyDense(in, in)),
		W2:    nn.NewParam(mat.NewEmptyDense(in, in)),
		W3:    nn.NewParam(mat.NewEmptyDense(in, in)),
		WCell: nn.NewParam(mat.NewEmptyDense(in, in)),
	}
}

type State struct {
	L1   ag.Node
	L2   ag.Node
	L3   ag.Node
	Cand ag.Node
	Cell ag.Node
	Y    ag.Node
}

type Processor struct {
	nn.BaseProcessor
	w1     ag.Node
	w2     ag.Node
	w3     ag.Node
	wCell  ag.Node
	States []*State
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
		w1:     g.NewWrap(m.W1),
		w2:     g.NewWrap(m.W2),
		w3:     g.NewWrap(m.W3),
		wCell:  g.NewWrap(m.WCell),
		States: nil,
	}
}

func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("ltm: the initial state must be set before any input")
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

func (p *Processor) LastState() *State {
	n := len(p.States)
	if n == 0 {
		return nil
	}
	return p.States[n-1]
}

// l1 = sigmoid(w1 (dot) (x + yPrev))
// l2 = sigmoid(w2 (dot) (x + yPrev))
// l3 = sigmoid(w3 (dot) (x + yPrev))
// c = l1 * l2 + cellPrev
// cell = sigmoid(c (dot) wCell + bCell)
// y = cell * l3
func (p *Processor) forward(x ag.Node) (s *State) {
	g := p.Graph
	s = new(State)
	yPrev, cellPrev := p.prev()
	h := x
	if yPrev != nil {
		h = g.Add(h, yPrev)
	}
	s.L1 = g.Sigmoid(g.Mul(p.w1, h))
	s.L2 = g.Sigmoid(g.Mul(p.w2, h))
	s.L3 = g.Sigmoid(g.Mul(p.w3, h))
	s.Cand = g.Prod(s.L1, s.L2)
	if cellPrev != nil {
		s.Cand = g.Add(s.Cand, cellPrev)
	}
	s.Cell = g.Sigmoid(g.Mul(p.wCell, s.Cand))
	s.Y = g.Prod(s.Cell, s.L3)
	return
}

func (p *Processor) prev() (yPrev, cellPrev ag.Node) {
	s := p.LastState()
	if s != nil {
		yPrev = s.Y
		cellPrev = s.Cell
	}
	return
}
