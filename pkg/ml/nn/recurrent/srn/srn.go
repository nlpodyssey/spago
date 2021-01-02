// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package srn

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
	W      nn.Param `spago:"type:weights"`
	WRec   nn.Param `spago:"type:weights"`
	B      nn.Param `spago:"type:biases"`
	States []*State `spago:"scope:processor"`
}

// State represent a state of the SRN recurrent network.
type State struct {
	Y ag.Node
}

// New returns a new model with parameters initialized to zeros.
func New(in, out int) *Model {
	return &Model{
		W:    nn.NewParam(mat.NewEmptyDense(out, in)),
		WRec: nn.NewParam(mat.NewEmptyDense(out, out)),
		B:    nn.NewParam(mat.NewEmptyVecDense(out)),
	}
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model) SetInitialState(state *State) {
	if len(m.States) > 0 {
		log.Fatal("srn: the initial state must be set before any input")
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

// y = tanh(w (dot) x + b + wRec (dot) yPrev)
func (m *Model) forward(x ag.Node) (s *State) {
	g := m.Graph()
	s = new(State)
	yPrev := m.prev()
	s.Y = g.Tanh(nn.Affine(g, m.B, m.W, x, m.WRec, yPrev))
	return
}

func (m *Model) prev() (yPrev ag.Node) {
	s := m.LastState()
	if s != nil {
		yPrev = s.Y
	}
	return
}
