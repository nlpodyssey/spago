// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package indrnn

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
)

var _ nn.Model = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.Module
	W          nn.Param[T]     `spago:"type:weights"`
	WRec       nn.Param[T]     `spago:"type:weights"`
	B          nn.Param[T]     `spago:"type:biases"`
	Activation activation.Name // output activation
}

// State represent a state of the IndRNN recurrent network.
type State[T mat.DType] struct {
	Y ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int, activation activation.Name) *Model[T] {
	return &Model[T]{
		W:          nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		WRec:       nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		B:          nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		Activation: activation,
	}
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
// y = f(w (dot) x + wRec * yPrev + b)
func (m *Model[T]) Next(state *State[T], x ag.Node[T]) (s *State[T]) {
	s = new(State[T])

	var yPrev ag.Node[T] = nil
	if state != nil {
		yPrev = state.Y
	}

	h := ag.Affine[T](m.B, m.W, x)
	if yPrev != nil {
		h = ag.Add(h, ag.Prod[T](m.WRec, yPrev))
	}
	s.Y = activation.Do(m.Activation, h)
	return
}
