// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package srn

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	W    nn.Param
	WRec nn.Param
	B    nn.Param
}

// State represent a state of the SRN recurrent network.
type State struct {
	Y ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, out int) *Model {
	return &Model{
		W:    nn.NewParam(mat.NewEmptyDense[T](out, in)),
		WRec: nn.NewParam(mat.NewEmptyDense[T](out, out)),
		B:    nn.NewParam(mat.NewEmptyVecDense[T](out)),
	}
}

func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	var s *State = nil
	for i, x := range xs {
		s = m.Next(s, x)
		ys[i] = s.Y
	}
	return ys
}

// Next performs a single forward step, producing a new state.
//
// y = tanh(w (dot) x + b + wRec (dot) yPrev)
func (m *Model) Next(state *State, x ag.Node) (s *State) {
	s = new(State)

	var yPrev ag.Node = nil
	if state != nil {
		yPrev = state.Y
	}

	s.Y = ag.Tanh(ag.Affine(m.B, m.W, x, m.WRec, yPrev))
	return
}
