// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package mist provides an implementation of the MIST (MIxed hiSTory) recurrent network as
described in "Analyzing and Exploiting NARX Recurrent Neural Networks for Long-Term Dependencies"
by Di Pietro et al., 2018 (https://arxiv.org/pdf/1702.07805.pdf).
*/
package mist

import (
	"encoding/gob"
	"math"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	Wx          nn.Param `spago:"type:weights"`
	Wh          nn.Param `spago:"type:weights"`
	B           nn.Param `spago:"type:biases"`
	Wax         nn.Param `spago:"type:weights"`
	Wah         nn.Param `spago:"type:weights"`
	Ba          nn.Param `spago:"type:biases"`
	Wrx         nn.Param `spago:"type:weights"`
	Wrh         nn.Param `spago:"type:weights"`
	Br          nn.Param `spago:"type:biases"`
	NumOfDelays int
}

// State represent a state of the MIST recurrent network.
type State struct {
	Y ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out, numOfDelays int) *Model {
	return &Model{
		Wx:          nn.NewParam(mat.NewEmptyDense[T](out, in)),
		Wh:          nn.NewParam(mat.NewEmptyDense[T](out, out)),
		B:           nn.NewParam(mat.NewEmptyVecDense[T](out)),
		Wax:         nn.NewParam(mat.NewEmptyDense[T](out, in)),
		Wah:         nn.NewParam(mat.NewEmptyDense[T](out, out)),
		Ba:          nn.NewParam(mat.NewEmptyVecDense[T](out)),
		Wrx:         nn.NewParam(mat.NewEmptyDense[T](out, in)),
		Wrh:         nn.NewParam(mat.NewEmptyDense[T](out, out)),
		Br:          nn.NewParam(mat.NewEmptyVecDense[T](out)),
		NumOfDelays: numOfDelays,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
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
func (m *Model) Next(states []*State, x ag.Node) (s *State) {
	s = new(State)

	var yPrev ag.Node = nil
	if states != nil {
		yPrev = states[len(states)-1].Y
	}

	a := ag.Softmax(ag.Affine(m.Ba, m.Wax, x, m.Wah, yPrev))
	r := ag.Sigmoid(ag.Affine(m.Br, m.Wrx, x, m.Wrh, yPrev))
	s.Y = ag.Tanh(ag.Affine(m.B, m.Wx, x, m.Wh, tryProd(r, m.weightHistory(states, a))))
	return
}

func (m *Model) weightHistory(states []*State, a ag.Node) ag.Node {
	var sum ag.Node
	n := len(states)
	for i := 0; i < m.NumOfDelays; i++ {
		k := int(math.Pow(2.0, float64(i))) // base-2 exponential delay
		if k <= n {
			sum = ag.Add(sum, ag.ProdScalar(states[n-k].Y, ag.AtVec(a, i)))
		}
	}
	return sum
}

// tryProd returns the product if 'a' and 'b' are not nil, otherwise nil
func tryProd(a, b ag.Node) ag.Node {
	if a != nil && b != nil {
		return ag.Prod(a, b)
	}
	return nil
}
