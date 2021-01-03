// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfn

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
	WIn     nn.Param `spago:"type:weights"`
	WInRec  nn.Param `spago:"type:weights"`
	BIn     nn.Param `spago:"type:biases"`
	WFor    nn.Param `spago:"type:weights"`
	WForRec nn.Param `spago:"type:weights"`
	BFor    nn.Param `spago:"type:biases"`
	WCand   nn.Param `spago:"type:weights"`
	States  []*State `spago:"scope:processor"`
}

// State represent a state of the CFN recurrent network.
type State struct {
	InG  ag.Node
	ForG ag.Node
	Cand ag.Node
	Y    ag.Node
}

// New returns a new model with parameters initialized to zeros.
func New(in, out int) *Model {
	m := &Model{}
	m.WIn, m.WInRec, m.BIn = newGateParams(in, out)
	m.WFor, m.WForRec, m.BFor = newGateParams(in, out)
	m.WCand = nn.NewParam(mat.NewEmptyDense(out, in))
	return m
}

func newGateParams(in, out int) (w, wRec, b nn.Param) {
	w = nn.NewParam(mat.NewEmptyDense(out, in))
	wRec = nn.NewParam(mat.NewEmptyDense(out, out))
	b = nn.NewParam(mat.NewEmptyVecDense(out))
	return
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model) SetInitialState(state *State) {
	if len(m.States) > 0 {
		log.Fatal("cfn: the initial state must be set before any input")
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

// inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
// forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
// c = f(wc (dot) x)
// y = inG * c + f(yPrev) * forG
func (m *Model) forward(x ag.Node) (s *State) {
	g := m.Graph()
	s = new(State)
	yPrev := m.prev()
	s.InG = g.Sigmoid(nn.Affine(g, m.BIn, m.WIn, x, m.WInRec, yPrev))
	s.ForG = g.Sigmoid(nn.Affine(g, m.BFor, m.WFor, x, m.WForRec, yPrev))
	s.Cand = g.Tanh(g.Mul(m.WCand, x))
	s.Y = g.Prod(s.InG, s.Cand)
	if yPrev != nil {
		s.Y = g.Add(s.Y, g.Prod(g.Tanh(yPrev), s.ForG))
	}
	return
}

func (m *Model) prev() (yPrev ag.Node) {
	s := m.LastState()
	if s != nil {
		yPrev = s.Y
	}
	return
}
