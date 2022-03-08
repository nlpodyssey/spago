// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deltarnn

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
	W     nn.Param[T] `spago:"type:weights"`
	WRec  nn.Param[T] `spago:"type:weights"`
	B     nn.Param[T] `spago:"type:biases"`
	BPart nn.Param[T] `spago:"type:biases"`
	Alpha nn.Param[T] `spago:"type:weights"`
	Beta1 nn.Param[T] `spago:"type:weights"`
	Beta2 nn.Param[T] `spago:"type:weights"`
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// State represent a state of the DeltaRNN recurrent network.
type State[T mat.DType] struct {
	D1 ag.Node[T]
	D2 ag.Node[T]
	C  ag.Node[T]
	P  ag.Node[T]
	Y  ag.Node[T]
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int) *Model[T] {
	return &Model[T]{
		W:     nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		WRec:  nn.NewParam[T](mat.NewEmptyDense[T](out, out)),
		B:     nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		BPart: nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		Alpha: nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		Beta1: nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		Beta2: nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
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
// d1 = beta1 * w (dot) x + beta2 * wRec (dot) yPrev
// d2 = alpha * w (dot) x * wRec (dot) yPrev
// c = tanh(d1 + d2 + bc)
// p = sigmoid(w (dot) x + bp)
// y = f(p * c + (1 - p) * yPrev)
func (m *Model[T]) Next(state *State[T], x ag.Node[T]) (s *State[T]) {
	s = new(State[T])

	var yPrev ag.Node[T] = nil
	if state != nil {
		yPrev = state.Y
	}

	wx := ag.Mul[T](m.W, x)
	if yPrev == nil {
		s.D1 = ag.Prod[T](m.Beta1, wx)
		s.C = ag.Tanh(ag.Add[T](s.D1, m.B))
		s.P = ag.Sigmoid(ag.Add[T](wx, m.BPart))
		s.Y = ag.Tanh(ag.Prod(s.P, s.C))
		return
	}
	wyRec := ag.Mul[T](m.WRec, yPrev)
	s.D1 = ag.Add(ag.Prod[T](m.Beta1, wx), ag.Prod[T](m.Beta2, wyRec))
	s.D2 = ag.Prod(ag.Prod[T](m.Alpha, wx), wyRec)
	s.C = ag.Tanh(ag.Add[T](ag.Add[T](s.D1, s.D2), m.B))
	s.P = ag.Sigmoid(ag.Add[T](wx, m.BPart))
	s.Y = ag.Tanh(ag.Add(ag.Prod(s.P, s.C), ag.Prod(ag.ReverseSub(s.P, m.Session.Graph().Constant(1.0)), yPrev)))
	return
}
