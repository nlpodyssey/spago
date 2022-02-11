// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tpr

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	WInS   nn.Param[T] `spago:"type:weights"`
	WInR   nn.Param[T] `spago:"type:weights"`
	WRecS  nn.Param[T] `spago:"type:weights"`
	WRecR  nn.Param[T] `spago:"type:weights"`
	BS     nn.Param[T] `spago:"type:biases"`
	BR     nn.Param[T] `spago:"type:biases"`
	S      nn.Param[T] `spago:"type:weights"`
	R      nn.Param[T] `spago:"type:weights"`
	States []*State[T] `spago:"scope:processor"`
}

// State represent a state of the TPR recurrent network.
type State[T mat.DType] struct {
	AR ag.Node[T]
	AS ag.Node[T]
	S  ag.Node[T]
	R  ag.Node[T]
	Y  ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, nSymbols, dSymbols, nRoles, dRoles int) *Model[T] {
	return &Model[T]{
		WInS:  nn.NewParam[T](mat.NewEmptyDense[T](nSymbols, in)),
		WInR:  nn.NewParam[T](mat.NewEmptyDense[T](nRoles, in)),
		WRecS: nn.NewParam[T](mat.NewEmptyDense[T](nSymbols, dRoles*dSymbols)),
		WRecR: nn.NewParam[T](mat.NewEmptyDense[T](nRoles, dRoles*dSymbols)),
		BS:    nn.NewParam[T](mat.NewEmptyVecDense[T](nSymbols)),
		BR:    nn.NewParam[T](mat.NewEmptyVecDense[T](nRoles)),
		S:     nn.NewParam[T](mat.NewEmptyDense[T](dSymbols, nSymbols)),
		R:     nn.NewParam[T](mat.NewEmptyDense[T](dRoles, nRoles)),
	}
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model[T]) SetInitialState(state *State[T]) {
	if len(m.States) > 0 {
		log.Fatal("tpr: the initial state must be set before any input")
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

// aR = Sigmoid(wInR (dot) x + bR + wRecR (dot) yPrev)
// aS = Sigmoid(wInS (dot) x + bS + wRecS (dot) yPrev)
// r = embR (dot) aR
// s = embS (dot) aS
// b = s (dot) rT
// y = vec(b)
func (m *Model[T]) forward(x ag.Node[T]) (st *State[T]) {
	sPrev := m.LastState()
	var yPrev ag.Node[T]
	if sPrev != nil {
		yPrev = sPrev.Y
	}
	st = new(State[T])
	g := m.Graph()
	st.AR = g.Sigmoid(nn.Affine[T](g, m.BR, m.WInR, x, m.WRecR, yPrev))
	st.AS = g.Sigmoid(nn.Affine[T](g, m.BS, m.WInS, x, m.WRecS, yPrev))
	st.R = g.Mul(m.R, st.AR)
	st.S = g.Mul(m.S, st.AS)
	b := g.Mul(st.S, g.T(st.R))
	st.Y = g.Vec(b)
	return
}
