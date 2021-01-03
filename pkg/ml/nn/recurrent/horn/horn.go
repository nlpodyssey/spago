// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package horn provides an implementation of Higher Order Recurrent Neural Networks (HORN).
package horn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	W      nn.Param   `spago:"type:weights"`
	WRec   []nn.Param `spago:"type:weights"`
	B      nn.Param   `spago:"type:biases"`
	States []*State   `spago:"scope:processor"`
}

// State represent a state of the Horn recurrent network.
type State struct {
	Y ag.Node
}

// New returns a new model with parameters initialized to zeros.
func New(in, out, order int) *Model {
	wRec := make([]nn.Param, order, order)
	for i := 0; i < order; i++ {
		wRec[i] = nn.NewParam(mat.NewEmptyDense(out, out))
	}
	return &Model{
		W:    nn.NewParam(mat.NewEmptyDense(out, in)),
		WRec: wRec,
		B:    nn.NewParam(mat.NewEmptyVecDense(out)),
	}
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model) SetInitialState(state *State) {
	if len(m.States) > 0 {
		log.Fatal("horn: the initial state must be set before any input")
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

func (m *Model) forward(x ag.Node) (s *State) {
	g := m.Graph()
	s = new(State)
	h := nn.Affine(g, append([]ag.Node{m.B, m.W, x}, m.feedback()...)...)
	s.Y = g.Tanh(h)
	return
}

func (m *Model) feedback() []ag.Node {
	g := m.Graph()
	var ys []ag.Node
	n := len(m.States)
	for i := 0; i < utils.MinInt(len(m.WRec), n); i++ {
		alpha := g.NewScalar(mat.Pow(0.6, mat.Float(i+1)))
		ys = append(ys, m.WRec[i], g.ProdScalar(m.States[n-1-i].Y, alpha))
	}
	return ys
}
