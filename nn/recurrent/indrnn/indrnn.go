// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package indrnn

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
	W          nn.Param[T] `spago:"type:weights"`
	WRec       nn.Param[T] `spago:"type:weights"`
	B          nn.Param[T] `spago:"type:biases"`
	Activation ag.OpName   // output activation
	States     []*State[T] `spago:"scope:processor"`
}

// State represent a state of the IndRNN recurrent network.
type State[T mat.DType] struct {
	Y ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int, activation ag.OpName) *Model[T] {
	return &Model[T]{
		W:          nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		WRec:       nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		B:          nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		Activation: activation,
	}
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model[T]) SetInitialState(state *State[T]) {
	if len(m.States) > 0 {
		log.Fatal("indrnn: the initial state must be set before any input")
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

// y = f(w (dot) x + wRec * yPrev + b)
func (m *Model[T]) forward(x ag.Node[T]) (s *State[T]) {
	s = new(State[T])
	yPrev := m.prev()
	h := ag.Affine[T](m.B, m.W, x)
	if yPrev != nil {
		h = ag.Add(h, ag.Prod[T](m.WRec, yPrev))
	}
	s.Y = ag.Invoke(m.Activation, h)
	return
}

func (m *Model[T]) prev() (yPrev ag.Node[T]) {
	s := m.LastState()
	if s != nil {
		yPrev = s.Y
	}
	return
}
