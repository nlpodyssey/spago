// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ltm

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	W1    nn.Param
	W2    nn.Param
	W3    nn.Param
	WCell nn.Param
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

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in int) *Model {
	return &Model{
		W1:    nn.NewParam(mat.NewEmptyDense[T](in, in)),
		W2:    nn.NewParam(mat.NewEmptyDense[T](in, in)),
		W3:    nn.NewParam(mat.NewEmptyDense[T](in, in)),
		WCell: nn.NewParam(mat.NewEmptyDense[T](in, in)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	var s *State = nil
	for i, x := range xs {
		s = m.Next(s, x)
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
func (m *Model) Next(state *State, x ag.Node) (s *State) {
	s = new(State)

	var yPrev, cellPrev ag.Node = nil, nil
	if state != nil {
		yPrev, cellPrev = state.Y, state.Cell
	}

	h := x
	if yPrev != nil {
		h = ag.Add(h, yPrev)
	}
	s.L1 = ag.Sigmoid(ag.Mul(m.W1, h))
	s.L2 = ag.Sigmoid(ag.Mul(m.W2, h))
	s.L3 = ag.Sigmoid(ag.Mul(m.W3, h))
	s.Cand = ag.Prod(s.L1, s.L2)
	if cellPrev != nil {
		s.Cand = ag.Add(s.Cand, cellPrev)
	}
	s.Cell = ag.Sigmoid(ag.Mul(m.WCell, s.Cand))
	s.Y = ag.Prod(s.Cell, s.L3)
	return
}
