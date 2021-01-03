// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fsmn

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

// Model implements a variant of the Feedforward Sequential Memory Networks
// (https://arxiv.org/pdf/1512.08301.pdf) where the neurons in the same hidden layer
// are independent of each other and they are connected across layers as in the IndRNN.
type Model struct {
	nn.BaseModel
	W      nn.Param   `spago:"type:weights"`
	WRec   nn.Param   `spago:"type:weights"`
	WS     []nn.Param `spago:"type:weights"` // coefficient vectors for scaling
	B      nn.Param   `spago:"type:biases"`
	Order  int
	States []*State `spago:"scope:processor"`
}

// New returns a new model with parameters initialized to zeros.
func New(in, out, order int) *Model {
	WS := make([]nn.Param, order, order)
	for i := 0; i < order; i++ {
		WS[i] = nn.NewParam(mat.NewEmptyVecDense(out))
	}
	return &Model{
		W:     nn.NewParam(mat.NewEmptyDense(out, in)),
		WRec:  nn.NewParam(mat.NewEmptyVecDense(out)),
		WS:    WS,
		B:     nn.NewParam(mat.NewEmptyVecDense(out)),
		Order: order,
	}
}

// State represent a state of the FSMN recurrent network.
type State struct {
	Y ag.Node
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model) SetInitialState(state *State) {
	if len(m.States) > 0 {
		log.Fatal("fsmn: the initial state must be set before any input")
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
	h := nn.Affine(g, m.B, m.W, x)
	if len(m.States) > 0 {
		h = g.Add(h, g.Prod(m.WRec, m.feedback()))
	}
	s.Y = g.ReLU(h)
	return
}

func (m *Model) feedback() ag.Node {
	g := m.Graph()
	var y ag.Node
	n := len(m.States)
	min := utils.MinInt(m.Order, n)
	for i := 0; i < min; i++ {
		scaled := g.Prod(m.WS[i], g.NewWrapNoGrad(m.States[n-1-i].Y))
		if y == nil {
			y = scaled
		} else {
			y = g.Add(y, scaled)
		}
	}
	return y
}
