// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gru

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
	WPart    nn.Param[T] `spago:"type:weights"`
	WPartRec nn.Param[T] `spago:"type:weights"`
	BPart    nn.Param[T] `spago:"type:biases"`
	WRes     nn.Param[T] `spago:"type:weights"`
	WResRec  nn.Param[T] `spago:"type:weights"`
	BRes     nn.Param[T] `spago:"type:biases"`
	WCand    nn.Param[T] `spago:"type:weights"`
	WCandRec nn.Param[T] `spago:"type:weights"`
	BCand    nn.Param[T] `spago:"type:biases"`
	States   []*State[T] `spago:"scope:processor"`
}

// State represent a state of the GRU recurrent network.
type State[T mat.DType] struct {
	R ag.Node[T]
	P ag.Node[T]
	C ag.Node[T]
	Y ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int) *Model[T] {
	m := &Model[T]{}
	m.WPart, m.WPartRec, m.BPart = newGateParams[T](in, out)
	m.WRes, m.WResRec, m.BRes = newGateParams[T](in, out)
	m.WCand, m.WCandRec, m.BCand = newGateParams[T](in, out)
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
		log.Fatal("gru: the initial state must be set before any input")
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

// r = sigmoid(wr (dot) x + br + wrRec (dot) yPrev)
// p = sigmoid(wp (dot) x + bp + wpRec (dot) yPrev)
// c = f(wc (dot) x + bc + wcRec (dot) (yPrev * r))
// y = p * c + (1 - p) * yPrev
func (m *Model[T]) forward(x ag.Node[T]) (s *State[T]) {
	s = new(State[T])
	yPrev := m.prev()
	s.R = ag.Sigmoid(ag.Affine[T](m.BRes, m.WRes, x, m.WResRec, yPrev))
	s.P = ag.Sigmoid(ag.Affine[T](m.BPart, m.WPart, x, m.WPartRec, yPrev))
	s.C = ag.Tanh(ag.Affine[T](m.BCand, m.WCand, x, m.WCandRec, tryProd(yPrev, s.R)))
	s.Y = ag.Prod(s.P, s.C)
	if yPrev != nil {
		s.Y = ag.Add(s.Y, ag.Prod(ag.ReverseSub(s.P, x.Graph().Constant(1.0)), yPrev))
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

// tryProd returns the product if 'a' il not nil, otherwise nil
func tryProd[T mat.DType](a, b ag.Node[T]) ag.Node[T] {
	if a != nil {
		return ag.Prod(a, b)
	}
	return nil
}
