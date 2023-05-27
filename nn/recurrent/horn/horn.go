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
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	W    *nn.Param
	WRec []*nn.Param
	B    *nn.Param
}

// State represent a state of the Horn recurrent network.
type State struct {
	Y ag.DualValue
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, out, order int) *Model {
	wRec := make([]*nn.Param, order)
	for i := 0; i < order; i++ {
		wRec[i] = nn.NewParam(mat.NewDense[T](mat.WithShape(out, out)))
	}
	return &Model{
		W:    nn.NewParam(mat.NewDense[T](mat.WithShape(out, in))),
		WRec: wRec,
		B:    nn.NewParam(mat.NewDense[T](mat.WithShape(out))),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.DualValue) []ag.DualValue {
	ys := make([]ag.DualValue, len(xs))
	states := make([]*State, 0)
	var s *State = nil
	for i, x := range xs {
		s = m.Next(states, x)
		states = append(states, s)
		ys[i] = s.Y
	}
	return ys
}

// Next performs a single forward step, producing a new state.
func (m *Model) Next(states []*State, x ag.DualValue) (s *State) {
	s = new(State)

	fb := m.feedback(states)

	h := ag.Affine(m.B, m.W, x, fb...)
	s.Y = ag.Tanh(h)
	return
}

func (m *Model) feedback(states []*State) []ag.DualValue {
	var ys []ag.DualValue
	n := len(states)
	for i := 0; i < min(len(m.WRec), n); i++ {
		alpha := m.WRec[i].Value().NewScalar(math.Pow(0.6, float64(i+1)))
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
