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
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
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
	States      []*State `spago:"scope:processor"`
}

// State represent a state of the MIST recurrent network.
type State struct {
	Y ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New(in, out, numOfDelays int) *Model {
	return &Model{
		Wx:          nn.NewParam(mat.NewEmptyDense(out, in)),
		Wh:          nn.NewParam(mat.NewEmptyDense(out, out)),
		B:           nn.NewParam(mat.NewEmptyVecDense(out)),
		Wax:         nn.NewParam(mat.NewEmptyDense(out, in)),
		Wah:         nn.NewParam(mat.NewEmptyDense(out, out)),
		Ba:          nn.NewParam(mat.NewEmptyVecDense(out)),
		Wrx:         nn.NewParam(mat.NewEmptyDense(out, in)),
		Wrh:         nn.NewParam(mat.NewEmptyDense(out, out)),
		Br:          nn.NewParam(mat.NewEmptyVecDense(out)),
		NumOfDelays: numOfDelays,
	}
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model) SetInitialState(state *State) {
	if len(m.States) > 0 {
		log.Fatal("mist: the initial state must be set before any input")
	}
	m.States = append(m.States, state)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		s := m.forward(x)
		m.States = append(m.States, s)
		ys[i] = s.Y
	}
	return ys
}

// LastState returns the last state of the recurrent network.
// It returns nil if there are no states.
func (m *Model) LastState() *State {
	n := len(m.States)
	if n == 0 {
		return nil
	}
	return m.States[n-1]
}

func (m *Model) forward(x ag.Node) (s *State) {
	g := m.Graph()
	s = new(State)
	yPrev := m.yPrev()
	a := g.Softmax(nn.Affine(g, m.Ba, m.Wax, x, m.Wah, yPrev))
	r := g.Sigmoid(nn.Affine(g, m.Br, m.Wrx, x, m.Wrh, yPrev)) // TODO: evaluate whether to calculate this only in case of previous states
	s.Y = g.Tanh(nn.Affine(g, m.B, m.Wx, x, m.Wh, m.tryProd(r, m.weightHistory(a))))
	return
}

func (m *Model) yPrev() ag.Node {
	var yPrev ag.Node
	s := m.LastState()
	if s != nil {
		yPrev = s.Y
	}
	return yPrev
}

func (m *Model) weightHistory(a ag.Node) ag.Node {
	g := m.Graph()
	var sum ag.Node
	n := len(m.States)
	for i := 0; i < m.NumOfDelays; i++ {
		k := int(mat.Pow(2.0, mat.Float(i))) // base-2 exponential delay
		if k <= n {
			sum = g.Add(sum, g.ProdScalar(m.States[n-k].Y, g.AtVec(a, i)))
		}
	}
	return sum
}

// tryProd returns the product if 'a' and 'b' are not nil, otherwise nil
func (m *Model) tryProd(a, b ag.Node) ag.Node {
	if a != nil && b != nil {
		return m.Graph().Prod(a, b)
	}
	return nil
}
