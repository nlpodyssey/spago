// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ltm

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	W1     nn.Param `spago:"type:weights"`
	W2     nn.Param `spago:"type:weights"`
	W3     nn.Param `spago:"type:weights"`
	WCell  nn.Param `spago:"type:weights"`
	States []*State `spago:"scope:processor"`
}

// State represent a state of the LTM recurrent network.
type State struct {
	L1   ag.Node
	L2   ag.Node
	L3   ag.Node
	Cand ag.Node
	Cell ag.Node
	Y    ag.Node
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

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model) SetInitialState(state *State) {
	if len(m.States) > 0 {
		log.Fatal("ltm: the initial state must be set before any input")
	}
	m.States = append(m.States, state)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		s := m.forward(x)
		m.States = append(m.States, s)
		ys[i] = s.Y
	}
	return ys
}

// LastState returns the last state of the recurrent network.
// It returns nil if there are no states.
func (m *Model) LastState() *State {
	n := len(m.States)
	if n == 0 {
		return nil
	}
	return m.States[n-1]
}

// l1 = sigmoid(w1 (dot) (x + yPrev))
// l2 = sigmoid(w2 (dot) (x + yPrev))
// l3 = sigmoid(w3 (dot) (x + yPrev))
// c = l1 * l2 + cellPrev
// cell = sigmoid(c (dot) wCell + bCell)
// y = cell * l3
func (m *Model) forward(x ag.Node) (s *State) {
	g := m.Graph()
	s = new(State)
	yPrev, cellPrev := m.prev()
	h := x
	if yPrev != nil {
		h = g.Add(h, yPrev)
	}
	s.L1 = g.Sigmoid(g.Mul(m.W1, h))
	s.L2 = g.Sigmoid(g.Mul(m.W2, h))
	s.L3 = g.Sigmoid(g.Mul(m.W3, h))
	s.Cand = g.Prod(s.L1, s.L2)
	if cellPrev != nil {
		s.Cand = g.Add(s.Cand, cellPrev)
	}
	s.Cell = g.Sigmoid(g.Mul(m.WCell, s.Cand))
	s.Y = g.Prod(s.Cell, s.L3)
	return
}

func (m *Model) prev() (yPrev, cellPrev ag.Node) {
	s := m.LastState()
	if s != nil {
		yPrev = s.Y
		cellPrev = s.Cell
	}
	return
}
