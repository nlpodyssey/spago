// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package horn provides an implementation of Higher Order Recurrent Neural Networks (HORN).
package horn

import (
	"encoding/gob"
	"math"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.Module
	W    nn.Param[T]   `spago:"type:weights"`
	WRec []nn.Param[T] `spago:"type:weights"`
	B    nn.Param[T]   `spago:"type:biases"`
}

// State represent a state of the Horn recurrent network.
type State[T mat.DType] struct {
	Y ag.Node
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out, order int) *Model[T] {
	wRec := make([]nn.Param[T], order)
	for i := 0; i < order; i++ {
		wRec[i] = nn.NewParam[T](mat.NewEmptyDense[T](out, out))
	}
	return &Model[T]{
		W:    nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		WRec: wRec,
		B:    nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	states := make([]*State[T], 0)
	var s *State[T] = nil
	for i, x := range xs {
		s = m.Next(states, x)
		states = append(states, s)
		ys[i] = s.Y
	}
	return ys
}

// Next performs a single forward step, producing a new state.
func (m *Model[T]) Next(states []*State[T], x ag.Node) (s *State[T]) {
	s = new(State[T])
	h := ag.Affine(append([]ag.Node{m.B, m.W, x}, m.feedback(states)...)...)
	s.Y = ag.Tanh(h)
	return
}

func (m *Model[T]) feedback(states []*State[T]) []ag.Node {
	var ys []ag.Node
	n := len(states)
	for i := 0; i < min(len(m.WRec), n); i++ {
		alpha := ag.NewScalar(m.WRec[i].Value().NewScalar(mat.Float(math.Pow(0.6, float64(i+1)))))
		ys = append(ys, m.WRec[i], ag.ProdScalar(states[n-1-i].Y, alpha))
	}
	return ys
}

// min returns the minimum value between a and b.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
