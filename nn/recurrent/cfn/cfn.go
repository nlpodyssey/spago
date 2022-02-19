// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfn

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
	WIn     nn.Param[T] `spago:"type:weights"`
	WInRec  nn.Param[T] `spago:"type:weights"`
	BIn     nn.Param[T] `spago:"type:biases"`
	WFor    nn.Param[T] `spago:"type:weights"`
	WForRec nn.Param[T] `spago:"type:weights"`
	BFor    nn.Param[T] `spago:"type:biases"`
	WCand   nn.Param[T] `spago:"type:weights"`
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
// inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
// forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
// c = f(wc (dot) x)
// y = inG * c + f(yPrev) * forG
func (m *Model[T]) Next(state *State[T], x ag.Node[T]) (s *State[T]) {
	s = new(State[T])

	var yPrev ag.Node[T] = nil
	if state != nil {
		yPrev = state.Y
	}

	s.InG = ag.Sigmoid(ag.Affine[T](m.BIn, m.WIn, x, m.WInRec, yPrev))
	s.ForG = ag.Sigmoid(ag.Affine[T](m.BFor, m.WFor, x, m.WForRec, yPrev))
	s.Cand = ag.Tanh(ag.Mul[T](m.WCand, x))
	s.Y = ag.Prod(s.InG, s.Cand)
	if yPrev != nil {
		s.Y = ag.Add(s.Y, ag.Prod(ag.Tanh(yPrev), s.ForG))
	}
	return
}
