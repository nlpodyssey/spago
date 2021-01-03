// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tpr

import (
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
	WInS   nn.Param `spago:"type:weights"`
	WInR   nn.Param `spago:"type:weights"`
	WRecS  nn.Param `spago:"type:weights"`
	WRecR  nn.Param `spago:"type:weights"`
	BS     nn.Param `spago:"type:biases"`
	BR     nn.Param `spago:"type:biases"`
	S      nn.Param `spago:"type:weights"`
	R      nn.Param `spago:"type:weights"`
	States []*State `spago:"scope:processor"`
}

// State represent a state of the TPR recurrent network.
type State struct {
	AR ag.Node
	AS ag.Node
	S  ag.Node
	R  ag.Node
	Y  ag.Node
}

// New returns a new model with parameters initialized to zeros.
func New(in, nSymbols, dSymbols, nRoles, dRoles int) *Model {
	return &Model{
		WInS:  nn.NewParam(mat.NewEmptyDense(nSymbols, in)),
		WInR:  nn.NewParam(mat.NewEmptyDense(nRoles, in)),
		WRecS: nn.NewParam(mat.NewEmptyDense(nSymbols, dRoles*dSymbols)),
		WRecR: nn.NewParam(mat.NewEmptyDense(nRoles, dRoles*dSymbols)),
		BS:    nn.NewParam(mat.NewEmptyVecDense(nSymbols)),
		BR:    nn.NewParam(mat.NewEmptyVecDense(nRoles)),
		S:     nn.NewParam(mat.NewEmptyDense(dSymbols, nSymbols)),
		R:     nn.NewParam(mat.NewEmptyDense(dRoles, nRoles)),
	}
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model) SetInitialState(state *State) {
	if len(m.States) > 0 {
		log.Fatal("tpr: the initial state must be set before any input")
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

// aR = Sigmoid(wInR (dot) x + bR + wRecR (dot) yPrev)
// aS = Sigmoid(wInS (dot) x + bS + wRecS (dot) yPrev)
// r = embR (dot) aR
// s = embS (dot) aS
// b = s (dot) rT
// y = vec(b)
func (m *Model) forward(x ag.Node) (st *State) {
	sPrev := m.LastState()
	var yPrev ag.Node
	if sPrev != nil {
		yPrev = sPrev.Y
	}
	st = new(State)
	g := m.Graph()
	st.AR = g.Sigmoid(nn.Affine(g, m.BR, m.WInR, x, m.WRecR, yPrev))
	st.AS = g.Sigmoid(nn.Affine(g, m.BS, m.WInS, x, m.WRecS, yPrev))
	st.R = g.Mul(m.R, st.AR)
	st.S = g.Mul(m.S, st.AS)
	b := g.Mul(st.S, g.T(st.R))
	st.Y = g.Vec(b)
	return
}
