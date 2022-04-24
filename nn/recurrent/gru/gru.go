// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gru

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
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

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	var s *State[T] = nil
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
func (m *Model[T]) Next(state *State[T], x ag.Node[T]) (s *State[T]) {
	s = new(State[T])

	var yPrev ag.Node[T] = nil
	if state != nil {
		yPrev = state.Y
	}

	s.R = ag.Sigmoid(ag.Affine[T](m.BRes, m.WRes, x, m.WResRec, yPrev))
	s.P = ag.Sigmoid(ag.Affine[T](m.BPart, m.WPart, x, m.WPartRec, yPrev))
	s.C = ag.Tanh(ag.Affine[T](m.BCand, m.WCand, x, m.WCandRec, tryProd(yPrev, s.R)))
	s.Y = ag.Prod(s.P, s.C)
	if yPrev != nil {
		s.Y = ag.Add(s.Y, ag.Prod(ag.ReverseSub(s.P, ag.Constant[T](1.0)), yPrev))
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
