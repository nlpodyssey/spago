// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ltm

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel
	W1    nn.Param[T] `spago:"type:weights"`
	W2    nn.Param[T] `spago:"type:weights"`
	W3    nn.Param[T] `spago:"type:weights"`
	WCell nn.Param[T] `spago:"type:weights"`
}

// State represent a state of the LTM recurrent network.
type State[T mat.DType] struct {
	L1   ag.Node[T]
	L2   ag.Node[T]
	L3   ag.Node[T]
	Cand ag.Node[T]
	Cell ag.Node[T]
	Y    ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in int) *Model[T] {
	return &Model[T]{
		W1:    nn.NewParam[T](mat.NewEmptyDense[T](in, in)),
		W2:    nn.NewParam[T](mat.NewEmptyDense[T](in, in)),
		W3:    nn.NewParam[T](mat.NewEmptyDense[T](in, in)),
		WCell: nn.NewParam[T](mat.NewEmptyDense[T](in, in)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	states := make([]*State[T], 0)
	var s *State[T] = nil
	for i, x := range xs {
		s = m.Next(s, x)
		states = append(states, s)
		ys[i] = s.Y
	}
	return ys
}

// Next performs a single forward step, producing a new state.
//
// l1 = sigmoid(w1 (dot) (x + yPrev))
// l2 = sigmoid(w2 (dot) (x + yPrev))
// l3 = sigmoid(w3 (dot) (x + yPrev))
// c = l1 * l2 + cellPrev
// cell = sigmoid(c (dot) wCell + bCell)
// y = cell * l3
func (m *Model[T]) Next(state *State[T], x ag.Node[T]) (s *State[T]) {
	s = new(State[T])

	var yPrev, cellPrev ag.Node[T] = nil, nil
	if state != nil {
		yPrev, cellPrev = state.Y, state.Cell
	}

	h := x
	if yPrev != nil {
		h = ag.Add(h, yPrev)
	}
	s.L1 = ag.Sigmoid(ag.Mul[T](m.W1, h))
	s.L2 = ag.Sigmoid(ag.Mul[T](m.W2, h))
	s.L3 = ag.Sigmoid(ag.Mul[T](m.W3, h))
	s.Cand = ag.Prod(s.L1, s.L2)
	if cellPrev != nil {
		s.Cand = ag.Add(s.Cand, cellPrev)
	}
	s.Cell = ag.Sigmoid(ag.Mul[T](m.WCell, s.Cand))
	s.Y = ag.Prod(s.Cell, s.L3)
	return
}
