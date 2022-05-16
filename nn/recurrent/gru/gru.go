// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gru

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
	WPart    nn.Param `spago:"type:weights"`
	WPartRec nn.Param `spago:"type:weights"`
	BPart    nn.Param `spago:"type:biases"`
	WRes     nn.Param `spago:"type:weights"`
	WResRec  nn.Param `spago:"type:weights"`
	BRes     nn.Param `spago:"type:biases"`
	WCand    nn.Param `spago:"type:weights"`
	WCandRec nn.Param `spago:"type:weights"`
	BCand    nn.Param `spago:"type:biases"`
}

// State represent a state of the GRU recurrent network.
type State struct {
	R ag.Node
	P ag.Node
	C ag.Node
	Y ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, out int) *Model {
	m := &Model{}
	m.WPart, m.WPartRec, m.BPart = newGateParams[T](in, out)
	m.WRes, m.WResRec, m.BRes = newGateParams[T](in, out)
	m.WCand, m.WCandRec, m.BCand = newGateParams[T](in, out)
	return m
}

func newGateParams[T float.DType](in, out int) (w, wRec, b nn.Param) {
	w = nn.NewParam(mat.NewEmptyDense[T](out, in))
	wRec = nn.NewParam(mat.NewEmptyDense[T](out, out))
	b = nn.NewParam(mat.NewEmptyVecDense[T](out))
	return
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
// r = sigmoid(wr (dot) x + br + wrRec (dot) yPrev)
// p = sigmoid(wp (dot) x + bp + wpRec (dot) yPrev)
// c = f(wc (dot) x + bc + wcRec (dot) (yPrev * r))
// y = p * c + (1 - p) * yPrev
func (m *Model) Next(state *State, x ag.Node) (s *State) {
	s = new(State)

	var yPrev ag.Node = nil
	if state != nil {
		yPrev = state.Y
	}

	s.R = ag.Sigmoid(ag.Affine(m.BRes, m.WRes, x, m.WResRec, yPrev))
	s.P = ag.Sigmoid(ag.Affine(m.BPart, m.WPart, x, m.WPartRec, yPrev))
	s.C = ag.Tanh(ag.Affine(m.BCand, m.WCand, x, m.WCandRec, tryProd(yPrev, s.R)))
	s.Y = ag.Prod(s.P, s.C)
	if yPrev != nil {
		one := ag.Var(x.Value().NewScalar(float.Float(1.0)))
		s.Y = ag.Add(s.Y, ag.Prod(ag.ReverseSub(s.P, one), yPrev))
	}
	return
}

// tryProd returns the product if 'a' il not nil, otherwise nil
func tryProd(a, b ag.Node) ag.Node {
	if a != nil {
		return ag.Prod(a, b)
	}
	return nil
}
