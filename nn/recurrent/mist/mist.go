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
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	Wx          *nn.Param
	Wh          *nn.Param
	B           *nn.Param
	Wax         *nn.Param
	Wah         *nn.Param
	Ba          *nn.Param
	Wrx         *nn.Param
	Wrh         *nn.Param
	Br          *nn.Param
	NumOfDelays int
}

// State represent a state of the MIST recurrent network.
type State struct {
	Y ag.DualValue
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, out, numOfDelays int) *Model {
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

	var yPrev ag.DualValue = nil
	if states != nil {
		yPrev = states[len(states)-1].Y
	}

	a := ag.Softmax(ag.Affine(m.Ba, m.Wax, x, m.Wah, yPrev))
	r := ag.Sigmoid(ag.Affine(m.Br, m.Wrx, x, m.Wrh, yPrev))
	s.Y = ag.Tanh(ag.Affine(m.B, m.Wx, x, m.Wh, tryProd(r, m.weightHistory(states, a))))
	return
}

func (m *Model) weightHistory(states []*State, a ag.DualValue) ag.DualValue {
	var sum ag.DualValue
	n := len(states)
	for i := 0; i < m.NumOfDelays; i++ {
		k := int(math.Pow(2.0, float64(i))) // base-2 exponential delay
		if k <= n {
			sum = ag.Add(sum, ag.ProdScalar(states[n-k].Y, ag.At(a, i)))
		}
	}
	return sum
}

// tryProd returns the product if 'a' and 'b' are not nil, otherwise nil
func tryProd(a, b ag.DualValue) ag.DualValue {
	if a != nil && b != nil {
		return ag.Prod(a, b)
	}
	return nil
}
