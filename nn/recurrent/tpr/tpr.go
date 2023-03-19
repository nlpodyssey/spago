// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tpr

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
	WInS  nn.Param
	WInR  nn.Param
	WRecS nn.Param
	WRecR nn.Param
	BS    nn.Param
	BR    nn.Param
	S     nn.Param
	R     nn.Param
}

// State represent a state of the TPR recurrent network.
type State struct {
	AR ag.Node
	AS ag.Node
	S  ag.Node
	R  ag.Node
	Y  ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, nSymbols, dSymbols, nRoles, dRoles int) *Model {
	return &Model{
		WInS:  nn.NewParam(mat.NewEmptyDense[T](nSymbols, in)),
		WInR:  nn.NewParam(mat.NewEmptyDense[T](nRoles, in)),
		WRecS: nn.NewParam(mat.NewEmptyDense[T](nSymbols, dRoles*dSymbols)),
		WRecR: nn.NewParam(mat.NewEmptyDense[T](nRoles, dRoles*dSymbols)),
		BS:    nn.NewParam(mat.NewEmptyVecDense[T](nSymbols)),
		BR:    nn.NewParam(mat.NewEmptyVecDense[T](nRoles)),
		S:     nn.NewParam(mat.NewEmptyDense[T](dSymbols, nSymbols)),
		R:     nn.NewParam(mat.NewEmptyDense[T](dRoles, nRoles)),
	}
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
// aR = Sigmoid(wInR (dot) x + bR + wRecR (dot) yPrev)
// aS = Sigmoid(wInS (dot) x + bS + wRecS (dot) yPrev)
// r = embR (dot) aR
// s = embS (dot) aS
// b = s (dot) rT
// y = vec(b)
func (m *Model) Next(state *State, x ag.Node) (st *State) {
	st = new(State)

	var yPrev ag.Node = nil
	if state != nil {
		yPrev = state.Y
	}

	st.AR = ag.Sigmoid(ag.Affine(m.BR, m.WInR, x, m.WRecR, yPrev))
	st.AS = ag.Sigmoid(ag.Affine(m.BS, m.WInS, x, m.WRecS, yPrev))
	st.R = ag.Mul(m.R, st.AR)
	st.S = ag.Mul(m.S, st.AS)
	b := ag.Mul(st.S, ag.T(st.R))
	st.Y = ag.T(ag.Flatten(b))
	return
}
