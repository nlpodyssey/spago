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

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	W     nn.Param `spago:"type:weights"`
	WRec  nn.Param `spago:"type:weights"`
	B     nn.Param `spago:"type:biases"`
	BPart nn.Param `spago:"type:biases"`
	Alpha nn.Param `spago:"type:weights"`
	Beta1 nn.Param `spago:"type:weights"`
	Beta2 nn.Param `spago:"type:weights"`
}

func init() {
	gob.Register(&Model{})
}

// State represent a state of the DeltaRNN recurrent network.
type State struct {
	D1 ag.Node
	D2 ag.Node
	C  ag.Node
	P  ag.Node
	Y  ag.Node
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int) *Model {
	return &Model{
		W:     nn.NewParam(mat.NewEmptyDense[T](out, in)),
		WRec:  nn.NewParam(mat.NewEmptyDense[T](out, out)),
		B:     nn.NewParam(mat.NewEmptyVecDense[T](out)),
		BPart: nn.NewParam(mat.NewEmptyVecDense[T](out)),
		Alpha: nn.NewParam(mat.NewEmptyVecDense[T](out)),
		Beta1: nn.NewParam(mat.NewEmptyVecDense[T](out)),
		Beta2: nn.NewParam(mat.NewEmptyVecDense[T](out)),
	}
}

// Forward performs the forward step for each input node and returns the result.
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
// d1 = beta1 * w (dot) x + beta2 * wRec (dot) yPrev
// d2 = alpha * w (dot) x * wRec (dot) yPrev
// c = tanh(d1 + d2 + bc)
// p = sigmoid(w (dot) x + bp)
// y = f(p * c + (1 - p) * yPrev)
func (m *Model) Next(state *State, x ag.Node) (s *State) {
	s = new(State)

	var yPrev ag.Node = nil
	if state != nil {
		yPrev = state.Y
	}

	wx := ag.Mul(m.W, x)
	if yPrev == nil {
		s.D1 = ag.Prod(m.Beta1, wx)
		s.C = ag.Tanh(ag.Add(s.D1, m.B))
		s.P = ag.Sigmoid(ag.Add(wx, m.BPart))
		s.Y = ag.Tanh(ag.Prod(s.P, s.C))
		return
	}
	wyRec := ag.Mul(m.WRec, yPrev)
	s.D1 = ag.Add(ag.Prod(m.Beta1, wx), ag.Prod(m.Beta2, wyRec))
	s.D2 = ag.Prod(ag.Prod(m.Alpha, wx), wyRec)
	s.C = ag.Tanh(ag.Add(ag.Add(s.D1, s.D2), m.B))
	s.P = ag.Sigmoid(ag.Add(wx, m.BPart))
	one := ag.Constant(s.P.Value().NewScalar(mat.Float(1.0)))
	s.Y = ag.Tanh(ag.Add(ag.Prod(s.P, s.C), ag.Prod(ag.ReverseSub(s.P, one), yPrev)))
	return
}
