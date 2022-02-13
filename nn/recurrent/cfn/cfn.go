// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfn

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"log"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	WIn     nn.Param[T] `spago:"type:weights"`
	WInRec  nn.Param[T] `spago:"type:weights"`
	BIn     nn.Param[T] `spago:"type:biases"`
	WFor    nn.Param[T] `spago:"type:weights"`
	WForRec nn.Param[T] `spago:"type:weights"`
	BFor    nn.Param[T] `spago:"type:biases"`
	WCand   nn.Param[T] `spago:"type:weights"`
	States  []*State[T] `spago:"scope:processor"`
}

// State represent a state of the CFN recurrent network.
type State[T mat.DType] struct {
	InG  ag.Node[T]
	ForG ag.Node[T]
	Cand ag.Node[T]
	Y    ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int) *Model[T] {
	m := &Model[T]{}
	m.WIn, m.WInRec, m.BIn = newGateParams[T](in, out)
	m.WFor, m.WForRec, m.BFor = newGateParams[T](in, out)
	m.WCand = nn.NewParam[T](mat.NewEmptyDense[T](out, in))
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
		log.Fatal("cfn: the initial state must be set before any input")
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

// inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
// forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
// c = f(wc (dot) x)
// y = inG * c + f(yPrev) * forG
func (m *Model[T]) forward(x ag.Node[T]) (s *State[T]) {
	g := m.Graph()
	s = new(State[T])
	yPrev := m.prev()
	s.InG = g.Sigmoid(g.Affine(m.BIn, m.WIn, x, m.WInRec, yPrev))
	s.ForG = g.Sigmoid(g.Affine(m.BFor, m.WFor, x, m.WForRec, yPrev))
	s.Cand = g.Tanh(g.Mul(m.WCand, x))
	s.Y = g.Prod(s.InG, s.Cand)
	if yPrev != nil {
		s.Y = g.Add(s.Y, g.Prod(g.Tanh(yPrev), s.ForG))
	}
	return
}

func (m *Model[T]) prev() (yPrev ag.Node[T]) {
	s := m.LastState()
	if s != nil {
		yPrev = s.Y
	}
	return
}
