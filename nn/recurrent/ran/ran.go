// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ran

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel
	WIn     nn.Param[T] `spago:"type:weights"`
	WInRec  nn.Param[T] `spago:"type:weights"`
	BIn     nn.Param[T] `spago:"type:biases"`
	WFor    nn.Param[T] `spago:"type:weights"`
	WForRec nn.Param[T] `spago:"type:weights"`
	BFor    nn.Param[T] `spago:"type:biases"`
	WCand   nn.Param[T] `spago:"type:weights"`
	BCand   nn.Param[T] `spago:"type:biases"`
}

// State represent a state of the RAN recurrent network.
type State[T mat.DType] struct {
	InG  ag.Node[T]
	ForG ag.Node[T]
	Cand ag.Node[T]
	C    ag.Node[T]
	Y    ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int) *Model[T] {
	m := &Model[T]{}
	m.WIn, m.WInRec, m.BIn = newGateParams[T](in, out)
	m.WFor, m.WForRec, m.BFor = newGateParams[T](in, out)
	m.WCand = nn.NewParam[T](mat.NewEmptyDense[T](out, in))
	m.BCand = nn.NewParam[T](mat.NewEmptyVecDense[T](out))
	return m
}

func newGateParams[T mat.DType](in, out int) (w, wRec, b nn.Param[T]) {
	w = nn.NewParam[T](mat.NewEmptyDense[T](out, in))
	wRec = nn.NewParam[T](mat.NewEmptyDense[T](out, out))
	b = nn.NewParam[T](mat.NewEmptyVecDense[T](out))
	return
}

func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	states := make([]*State[T], 0)
	var s *State[T] = nil
	for i, x := range xs {
		s = m.Next(s, x)
		states = append(states, s)
		ys[i] = s.Y
	}
	return ys
}

// Next performs a single forward step, producing a new state.
//
// inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
// forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
// cand = wc (dot) x + bc
// c = inG * c + forG * cPrev
// y = f(c)
func (m *Model[T]) Next(state *State[T], x ag.Node[T]) (s *State[T]) {
	s = new(State[T])

	var yPrev, cPrev ag.Node[T] = nil, nil
	if state != nil {
		yPrev, cPrev = state.Y, state.C
	}

	s.InG = ag.Sigmoid(ag.Affine[T](m.BIn, m.WIn, x, m.WInRec, yPrev))
	s.ForG = ag.Sigmoid(ag.Affine[T](m.BFor, m.WFor, x, m.WForRec, yPrev))
	s.Cand = ag.Affine[T](m.BCand, m.WCand, x)
	s.C = ag.Prod(s.InG, s.Cand)
	if cPrev != nil {
		s.C = ag.Add(s.C, ag.Prod(s.ForG, cPrev))
	}
	s.Y = ag.Tanh(s.C)
	return
}

// Importance returns the "importance" score for each element of the processed sequence.
func (m *Model[T]) Importance(states []*State[T]) [][]T {
	importance := make([][]T, len(states))
	for i := range importance {
		importance[i] = m.scores(states, i)
	}
	return importance
}

// importance computes the importance score of the previous states respect to the i-state.
// The output contains the importance score for each k-previous states.
func (m *Model[T]) scores(states []*State[T], i int) []T {
	scores := make([]T, len(states))
	incForgetProd := states[i].ForG.Value().Clone()
	for k := i; k >= 0; k-- {
		inG := states[k].InG.Value()
		forG := states[k].ForG.Value()
		scores[k] = inG.Prod(incForgetProd).Max()
		if k > 0 {
			incForgetProd.ProdInPlace(forG)
		}
	}
	return scores
}
