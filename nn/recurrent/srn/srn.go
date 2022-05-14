// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package srn

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.Module
	W    nn.Param[T] `spago:"type:weights"`
	WRec nn.Param[T] `spago:"type:weights"`
	B    nn.Param[T] `spago:"type:biases"`
}

// State represent a state of the SRN recurrent network.
type State[T mat.DType] struct {
	Y ag.Node
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int) *Model[T] {
	return &Model[T]{
		W:    nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		WRec: nn.NewParam[T](mat.NewEmptyDense[T](out, out)),
		B:    nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
	}
}

func (m *Model[T]) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	var s *State[T] = nil
	for i, x := range xs {
		s = m.Next(s, x)
		ys[i] = s.Y
	}
	return ys
}

// Next performs a single forward step, producing a new state.
//
// y = tanh(w (dot) x + b + wRec (dot) yPrev)
func (m *Model[T]) Next(state *State[T], x ag.Node) (s *State[T]) {
	s = new(State[T])

	var yPrev ag.Node = nil
	if state != nil {
		yPrev = state.Y
	}

	s.Y = ag.Tanh(ag.Affine(m.B, m.W, x, m.WRec, yPrev))
	return
}
