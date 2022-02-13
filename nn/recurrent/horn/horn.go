// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package horn provides an implementation of Higher Order Recurrent Neural Networks (HORN).
package horn

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/utils"
	"log"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	W      nn.Param[T]   `spago:"type:weights"`
	WRec   []nn.Param[T] `spago:"type:weights"`
	B      nn.Param[T]   `spago:"type:biases"`
	States []*State[T]   `spago:"scope:processor"`
}

// State represent a state of the Horn recurrent network.
type State[T mat.DType] struct {
	Y ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out, order int) *Model[T] {
	wRec := make([]nn.Param[T], order, order)
	for i := 0; i < order; i++ {
		wRec[i] = nn.NewParam[T](mat.NewEmptyDense[T](out, out))
	}
	return &Model[T]{
		W:    nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		WRec: wRec,
		B:    nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
	}
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model[T]) SetInitialState(state *State[T]) {
	if len(m.States) > 0 {
		log.Fatal("horn: the initial state must be set before any input")
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

func (m *Model[T]) forward(x ag.Node[T]) (s *State[T]) {
	g := m.Graph()
	s = new(State[T])
	h := g.Affine(append([]ag.Node[T]{m.B, m.W, x}, m.feedback()...)...)
	s.Y = g.Tanh(h)
	return
}

func (m *Model[T]) feedback() []ag.Node[T] {
	g := m.Graph()
	var ys []ag.Node[T]
	n := len(m.States)
	for i := 0; i < utils.MinInt(len(m.WRec), n); i++ {
		alpha := g.NewScalar(mat.Pow(0.6, T(i+1)))
		ys = append(ys, m.WRec[i], g.ProdScalar(m.States[n-1-i].Y, alpha))
	}
	return ys
}
