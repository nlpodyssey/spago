// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tpr

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
	WInS  nn.Param[T] `spago:"type:weights"`
	WInR  nn.Param[T] `spago:"type:weights"`
	WRecS nn.Param[T] `spago:"type:weights"`
	WRecR nn.Param[T] `spago:"type:weights"`
	BS    nn.Param[T] `spago:"type:biases"`
	BR    nn.Param[T] `spago:"type:biases"`
	S     nn.Param[T] `spago:"type:weights"`
	R     nn.Param[T] `spago:"type:weights"`
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

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	var s *State[T] = nil
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
func (m *Model[T]) Next(state *State[T], x ag.Node[T]) (st *State[T]) {
	st = new(State[T])

	var yPrev ag.Node[T] = nil
	if state != nil {
		yPrev = state.Y
	}

	st.AR = ag.Sigmoid(ag.Affine[T](m.BR, m.WInR, x, m.WRecR, yPrev))
	st.AS = ag.Sigmoid(ag.Affine[T](m.BS, m.WInS, x, m.WRecS, yPrev))
	st.R = ag.Mul[T](m.R, st.AR)
	st.S = ag.Mul[T](m.S, st.AS)
	b := ag.Mul(st.S, ag.T(st.R))
	st.Y = ag.T(ag.Flatten(b))
	return
}
