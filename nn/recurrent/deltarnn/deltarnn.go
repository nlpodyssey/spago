// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deltarnn

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"log"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	W      nn.Param[T] `spago:"type:weights"`
	WRec   nn.Param[T] `spago:"type:weights"`
	B      nn.Param[T] `spago:"type:biases"`
	BPart  nn.Param[T] `spago:"type:biases"`
	Alpha  nn.Param[T] `spago:"type:weights"`
	Beta1  nn.Param[T] `spago:"type:weights"`
	Beta2  nn.Param[T] `spago:"type:weights"`
	States []*State[T] `spago:"scope:processor"`
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

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model[T]) SetInitialState(state *State[T]) {
	if len(m.States) > 0 {
		log.Fatal("deltarnn: the initial state must be set before any input")
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

// d1 = beta1 * w (dot) x + beta2 * wRec (dot) yPrev
// d2 = alpha * w (dot) x * wRec (dot) yPrev
// c = tanh(d1 + d2 + bc)
// p = sigmoid(w (dot) x + bp)
// y = f(p * c + (1 - p) * yPrev)
func (m *Model[T]) forward(x ag.Node[T]) (s *State[T]) {
	g := m.Graph()
	s = new(State[T])
	yPrev := m.prev()
	wx := g.Mul(m.W, x)
	if yPrev == nil {
		s.D1 = g.Prod(m.Beta1, wx)
		s.C = g.Tanh(g.Add(s.D1, m.B))
		s.P = g.Sigmoid(g.Add(wx, m.BPart))
		s.Y = g.Tanh(g.Prod(s.P, s.C))
	} else {
		wyRec := g.Mul(m.WRec, yPrev)
		s.D1 = g.Add(g.Prod(m.Beta1, wx), g.Prod(m.Beta2, wyRec))
		s.D2 = g.Prod(g.Prod(m.Alpha, wx), wyRec)
		s.C = g.Tanh(g.Add(g.Add(s.D1, s.D2), m.B))
		s.P = g.Sigmoid(g.Add(wx, m.BPart))
		s.Y = g.Tanh(g.Add(g.Prod(s.P, s.C), g.Prod(g.ReverseSub(s.P, g.NewScalar(1.0)), yPrev)))
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
