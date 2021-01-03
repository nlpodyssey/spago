// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lstm

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
	UseRefinedGates bool
	WIn             nn.Param `spago:"type:weights"`
	WInRec          nn.Param `spago:"type:weights"`
	BIn             nn.Param `spago:"type:biases"`
	WOut            nn.Param `spago:"type:weights"`
	WOutRec         nn.Param `spago:"type:weights"`
	BOut            nn.Param `spago:"type:biases"`
	WFor            nn.Param `spago:"type:weights"`
	WForRec         nn.Param `spago:"type:weights"`
	BFor            nn.Param `spago:"type:biases"`
	WCand           nn.Param `spago:"type:weights"`
	WCandRec        nn.Param `spago:"type:weights"`
	BCand           nn.Param `spago:"type:biases"`
	States          []*State `spago:"scope:processor"`
}

// State represent a state of the LSTM recurrent network.
type State struct {
	InG  ag.Node
	OutG ag.Node
	ForG ag.Node
	Cand ag.Node
	Cell ag.Node
	Y    ag.Node
}

// Option allows to configure a new Model with your specific needs.
type Option func(*Model)

// SetRefinedGates sets whether to use refined gates.
// Refined Gate: A Simple and Effective Gating Mechanism for Recurrent Units
// (https://arxiv.org/pdf/2002.11338.pdf)
// TODO: panic input size and output size are different
func SetRefinedGates(value bool) Option {
	return func(m *Model) {
		m.UseRefinedGates = value
	}
}

// New returns a new model with parameters initialized to zeros.
func New(in, out int, options ...Option) *Model {
	m := &Model{}
	m.WIn, m.WInRec, m.BIn = newGateParams(in, out)
	m.WOut, m.WOutRec, m.BOut = newGateParams(in, out)
	m.WFor, m.WForRec, m.BFor = newGateParams(in, out)
	m.WCand, m.WCandRec, m.BCand = newGateParams(in, out)
	m.UseRefinedGates = false

	for _, option := range options {
		option(m)
	}
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
		log.Fatal("lstm: the initial state must be set before any input")
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

// forward computes the results with the following equations:
// inG = sigmoid(wIn (dot) x + bIn + wInRec (dot) yPrev)
// outG = sigmoid(wOut (dot) x + bOut + wOutRec (dot) yPrev)
// forG = sigmoid(wFor (dot) x + bFor + wForRec (dot) yPrev)
// cand = f(wCand (dot) x + bC + wCandRec (dot) yPrev)
// cell = inG * cand + forG * cellPrev
// y = outG * f(cell)
func (m *Model) forward(x ag.Node) (s *State) {
	g := m.Graph()
	s = new(State)
	yPrev, cellPrev := m.prev()
	s.InG = g.Sigmoid(nn.Affine(g, m.BIn, m.WIn, x, m.WInRec, yPrev))
	s.OutG = g.Sigmoid(nn.Affine(g, m.BOut, m.WOut, x, m.WOutRec, yPrev))
	s.ForG = g.Sigmoid(nn.Affine(g, m.BFor, m.WFor, x, m.WForRec, yPrev))
	s.Cand = g.Tanh(nn.Affine(g, m.BCand, m.WCand, x, m.WCandRec, yPrev))

	if m.UseRefinedGates {
		s.InG = g.Prod(s.InG, x)
		s.OutG = g.Prod(s.OutG, x)
	}

	if cellPrev != nil {
		s.Cell = g.Add(g.Prod(s.InG, s.Cand), g.Prod(s.ForG, cellPrev))
	} else {
		s.Cell = g.Prod(s.InG, s.Cand)
	}
	s.Y = g.Prod(s.OutG, g.Tanh(s.Cell))
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
