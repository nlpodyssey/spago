// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lstm

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	UseRefinedGates bool
	WIn             nn.Param[T] `spago:"type:weights"`
	WInRec          nn.Param[T] `spago:"type:weights"`
	BIn             nn.Param[T] `spago:"type:biases"`
	WOut            nn.Param[T] `spago:"type:weights"`
	WOutRec         nn.Param[T] `spago:"type:weights"`
	BOut            nn.Param[T] `spago:"type:biases"`
	WFor            nn.Param[T] `spago:"type:weights"`
	WForRec         nn.Param[T] `spago:"type:weights"`
	BFor            nn.Param[T] `spago:"type:biases"`
	WCand           nn.Param[T] `spago:"type:weights"`
	WCandRec        nn.Param[T] `spago:"type:weights"`
	BCand           nn.Param[T] `spago:"type:biases"`
	States          []*State[T] `spago:"scope:processor"`
}

// State represent a state of the LSTM recurrent network.
type State[T mat.DType] struct {
	InG  ag.Node[T]
	OutG ag.Node[T]
	ForG ag.Node[T]
	Cand ag.Node[T]
	Cell ag.Node[T]
	Y    ag.Node[T]
}

// Option allows to configure a new Model with your specific needs.
type Option[T mat.DType] func(*Model[T])

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// SetRefinedGates sets whether to use refined gates.
// Refined Gate: A Simple and Effective Gating Mechanism for Recurrent Units
// (https://arxiv.org/pdf/2002.11338.pdf)
// TODO: panic input size and output size are different
func SetRefinedGates[T mat.DType](value bool) Option[T] {
	return func(m *Model[T]) {
		m.UseRefinedGates = value
	}
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int, options ...Option[T]) *Model[T] {
	m := &Model[T]{}
	m.WIn, m.WInRec, m.BIn = newGateParams[T](in, out)
	m.WOut, m.WOutRec, m.BOut = newGateParams[T](in, out)
	m.WFor, m.WForRec, m.BFor = newGateParams[T](in, out)
	m.WCand, m.WCandRec, m.BCand = newGateParams[T](in, out)
	m.UseRefinedGates = false

	for _, option := range options {
		option(m)
	}
	return m
}

func newGateParams[T mat.DType](in, out int) (w, wRec, b nn.Param[T]) {
	w = nn.NewParam[T](mat.NewEmptyDense[T](out, in))
	wRec = nn.NewParam[T](mat.NewEmptyDense[T](out, out))
	b = nn.NewParam[T](mat.NewEmptyVecDense[T](out))
	return
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model[T]) SetInitialState(state *State[T]) {
	if len(m.States) > 0 {
		log.Fatal("lstm: the initial state must be set before any input")
	}
	m.States = append(m.States, state)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		s := m.forward(x)
		m.States = append(m.States, s)
		ys[i] = s.Y
	}
	return ys
}

// LastState returns the last state of the recurrent network.
// It returns nil if there are no states.
func (m *Model[T]) LastState() *State[T] {
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
func (m *Model[T]) forward(x ag.Node[T]) (s *State[T]) {
	g := m.Graph()
	s = new(State[T])
	yPrev, cellPrev := m.prev()
	s.InG = g.Sigmoid(nn.Affine[T](g, m.BIn, m.WIn, x, m.WInRec, yPrev))
	s.OutG = g.Sigmoid(nn.Affine[T](g, m.BOut, m.WOut, x, m.WOutRec, yPrev))
	s.ForG = g.Sigmoid(nn.Affine[T](g, m.BFor, m.WFor, x, m.WForRec, yPrev))
	s.Cand = g.Tanh(nn.Affine[T](g, m.BCand, m.WCand, x, m.WCandRec, yPrev))

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

func (m *Model[T]) prev() (yPrev, cellPrev ag.Node[T]) {
	s := m.LastState()
	if s != nil {
		yPrev = s.Y
		cellPrev = s.Cell
	}
	return
}
