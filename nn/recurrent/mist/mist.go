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

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.Module[T]
	Wx          nn.Param[T] `spago:"type:weights"`
	Wh          nn.Param[T] `spago:"type:weights"`
	B           nn.Param[T] `spago:"type:biases"`
	Wax         nn.Param[T] `spago:"type:weights"`
	Wah         nn.Param[T] `spago:"type:weights"`
	Ba          nn.Param[T] `spago:"type:biases"`
	Wrx         nn.Param[T] `spago:"type:weights"`
	Wrh         nn.Param[T] `spago:"type:weights"`
	Br          nn.Param[T] `spago:"type:biases"`
	NumOfDelays int
}

// State represent a state of the MIST recurrent network.
type State[T mat.DType] struct {
	Y ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out, numOfDelays int) *Model[T] {
	return &Model[T]{
		Wx:          nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		Wh:          nn.NewParam[T](mat.NewEmptyDense[T](out, out)),
		B:           nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		Wax:         nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		Wah:         nn.NewParam[T](mat.NewEmptyDense[T](out, out)),
		Ba:          nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		Wrx:         nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		Wrh:         nn.NewParam[T](mat.NewEmptyDense[T](out, out)),
		Br:          nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
		NumOfDelays: numOfDelays,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
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
func (m *Model[T]) Next(states []*State[T], x ag.Node[T]) (s *State[T]) {
	s = new(State[T])

	var yPrev ag.Node[T] = nil
	if states != nil {
		yPrev = states[len(states)-1].Y
	}

	a := ag.Softmax(ag.Affine[T](m.Ba, m.Wax, x, m.Wah, yPrev))
	r := ag.Sigmoid(ag.Affine[T](m.Br, m.Wrx, x, m.Wrh, yPrev))
	s.Y = ag.Tanh(ag.Affine[T](m.B, m.Wx, x, m.Wh, tryProd[T](r, m.weightHistory(states, a))))
	return
}

func (m *Model[T]) weightHistory(states []*State[T], a ag.Node[T]) ag.Node[T] {
	var sum ag.Node[T]
	n := len(states)
	for i := 0; i < m.NumOfDelays; i++ {
		k := int(mat.Pow(2.0, T(i))) // base-2 exponential delay
		if k <= n {
			sum = ag.Add(sum, ag.ProdScalar(states[n-k].Y, ag.AtVec(a, i)))
		}
	}
	return sum
}

// tryProd returns the product if 'a' and 'b' are not nil, otherwise nil
func tryProd[T mat.DType](a, b ag.Node[T]) ag.Node[T] {
	if a != nil && b != nil {
		return ag.Prod(a, b)
	}
	return nil
}
