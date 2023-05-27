// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package indrnn

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	W          *nn.Param
	WRec       *nn.Param
	B          *nn.Param
	Activation activation.Name // output activation
}

// State represent a state of the IndRNN recurrent network.
type State struct {
	Y ag.DualValue
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, out int, activation activation.Name) *Model {
	return &Model{
		W:          nn.NewParam(mat.NewDense[T](mat.WithShape(out, in))),
		WRec:       nn.NewParam(mat.NewDense[T](mat.WithShape(out))),
		B:          nn.NewParam(mat.NewDense[T](mat.WithShape(out))),
		Activation: activation,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.DualValue) []ag.DualValue {
	ys := make([]ag.DualValue, len(xs))
	var s *State = nil
	for i, x := range xs {
		s = m.Next(s, x)
		ys[i] = s.Y
	}
	return ys
}

// Next performs a single forward step, producing a new state.
//
// y = f(w (dot) x + wRec * yPrev + b)
func (m *Model) Next(state *State, x ag.DualValue) (s *State) {
	s = new(State)

	var yPrev ag.DualValue = nil
	if state != nil {
		yPrev = state.Y
	}

	h := ag.Affine(m.B, m.W, x)
	if yPrev != nil {
		h = ag.Add(h, ag.Prod(m.WRec, yPrev))
	}
	s.Y = activation.Do(m.Activation, h)
	return
}
