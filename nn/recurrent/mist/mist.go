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
	"log"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
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
	States      []*State[T] `spago:"scope:processor"`
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

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model[T]) SetInitialState(state *State[T]) {
	if len(m.States) > 0 {
		log.Fatal("mist: the initial state must be set before any input")
	}
	m.States = append(m.States, state)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		s := m.forward(x)
		m.States = append(m.States, s)
		ys[i] = s.Y
	}
	return ys
}

// LastState returns the last state of the recurrent network.
// It returns nil if there are no states.
func (m *Model[T]) LastState() *State[T] {
	n := len(m.States)
	if n == 0 {
		return nil
	}
	return m.States[n-1]
}

func (m *Model[T]) forward(x ag.Node[T]) (s *State[T]) {
	g := m.Graph()
	s = new(State[T])
	yPrev := m.yPrev()
	a := g.Softmax(g.Affine(m.Ba, m.Wax, x, m.Wah, yPrev))
	r := g.Sigmoid(g.Affine(m.Br, m.Wrx, x, m.Wrh, yPrev)) // TODO: evaluate whether to calculate this only in case of previous states
	s.Y = g.Tanh(g.Affine(m.B, m.Wx, x, m.Wh, m.tryProd(r, m.weightHistory(a))))
	return
}

func (m *Model[T]) yPrev() ag.Node[T] {
	var yPrev ag.Node[T]
	s := m.LastState()
	if s != nil {
		yPrev = s.Y
	}
	return yPrev
}

func (m *Model[T]) weightHistory(a ag.Node[T]) ag.Node[T] {
	g := m.Graph()
	var sum ag.Node[T]
	n := len(m.States)
	for i := 0; i < m.NumOfDelays; i++ {
		k := int(mat.Pow(2.0, T(i))) // base-2 exponential delay
		if k <= n {
			sum = g.Add(sum, g.ProdScalar(m.States[n-k].Y, g.AtVec(a, i)))
		}
	}
	return sum
}

// tryProd returns the product if 'a' and 'b' are not nil, otherwise nil
func (m *Model[T]) tryProd(a, b ag.Node[T]) ag.Node[T] {
	if a != nil && b != nil {
		return m.Graph().Prod(a, b)
	}
	return nil
}
