// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfn

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
	WIn     nn.Param `spago:"type:weights"`
	WInRec  nn.Param `spago:"type:weights"`
	BIn     nn.Param `spago:"type:biases"`
	WFor    nn.Param `spago:"type:weights"`
	WForRec nn.Param `spago:"type:weights"`
	BFor    nn.Param `spago:"type:biases"`
	WCand   nn.Param `spago:"type:weights"`
}

// State represent a state of the CFN recurrent network.
type State struct {
	InG  ag.Node
	ForG ag.Node
	Cand ag.Node
	Y    ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, out int) *Model {
	m := &Model{}
	m.WIn, m.WInRec, m.BIn = newGateParams[T](in, out)
	m.WFor, m.WForRec, m.BFor = newGateParams[T](in, out)
	m.WCand = nn.NewParam(mat.NewEmptyDense[T](out, in))
	return m
}

func newGateParams[T float.DType](in, out int) (w, wRec, b nn.Param) {
	w = nn.NewParam(mat.NewEmptyDense[T](out, in))
	wRec = nn.NewParam(mat.NewEmptyDense[T](out, out))
	b = nn.NewParam(mat.NewEmptyVecDense[T](out))
	return
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
// inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
// forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
// c = f(wc (dot) x)
// y = inG * c + f(yPrev) * forG
func (m *Model) Next(state *State, x ag.Node) (s *State) {
	s = new(State)

	var yPrev ag.Node = nil
	if state != nil {
		yPrev = state.Y
	}

	s.InG = ag.Sigmoid(ag.Affine(m.BIn, m.WIn, x, m.WInRec, yPrev))
	s.ForG = ag.Sigmoid(ag.Affine(m.BFor, m.WFor, x, m.WForRec, yPrev))
	s.Cand = ag.Tanh(ag.Mul(m.WCand, x))
	s.Y = ag.Prod(s.InG, s.Cand)
	if yPrev != nil {
		s.Y = ag.Add(s.Y, ag.Prod(ag.Tanh(yPrev), s.ForG))
	}
	return
}
