// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ran

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
	WIn     *nn.Param
	WInRec  *nn.Param
	BIn     *nn.Param
	WFor    *nn.Param
	WForRec *nn.Param
	BFor    *nn.Param
	WCand   *nn.Param
	BCand   *nn.Param
}

// State represent a state of the RAN recurrent network.
type State struct {
	InG  ag.Node
	ForG ag.Node
	Cand ag.Node
	C    ag.Node
	Y    ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, out int) *Model {
	m := &Model{}
	m.WIn, m.WInRec, m.BIn = newGateParams[T](in, out)
	m.WFor, m.WForRec, m.BFor = newGateParams[T](in, out)
	m.WCand = nn.NewParam(mat.NewEmptyDense[T](out, in))
	m.BCand = nn.NewParam(mat.NewEmptyVecDense[T](out))
	return m
}

func newGateParams[T float.DType](in, out int) (w, wRec, b *nn.Param) {
	w = nn.NewParam(mat.NewEmptyDense[T](out, in))
	wRec = nn.NewParam(mat.NewEmptyDense[T](out, out))
	b = nn.NewParam(mat.NewEmptyVecDense[T](out))
	return
}

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
// inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
// forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
// cand = wc (dot) x + bc
// c = inG * c + forG * cPrev
// y = f(c)
func (m *Model) Next(state *State, x ag.Node) (s *State) {
	s = new(State)

	var yPrev, cPrev ag.Node = nil, nil
	if state != nil {
		yPrev, cPrev = state.Y, state.C
	}

	s.InG = ag.Sigmoid(ag.Affine(m.BIn, m.WIn, x, m.WInRec, yPrev))
	s.ForG = ag.Sigmoid(ag.Affine(m.BFor, m.WFor, x, m.WForRec, yPrev))
	s.Cand = ag.Affine(m.BCand, m.WCand, x)
	s.C = ag.Prod(s.InG, s.Cand)
	if cPrev != nil {
		s.C = ag.Add(s.C, ag.Prod(s.ForG, cPrev))
	}
	s.Y = ag.Tanh(s.C)
	return
}

// Importance returns the "importance" score for each element of the processed sequence.
func (m *Model) Importance(states []*State) [][]float64 {
	importance := make([][]float64, len(states))
	for i := range importance {
		importance[i] = m.scores(states, i)
	}
	return importance
}

// importance computes the importance score of the previous states respect to the i-state.
// The output contains the importance score for each k-previous states.
func (m *Model) scores(states []*State, i int) []float64 {
	scores := make([]float64, len(states))
	incForgetProd := states[i].ForG.Value().Clone()
	for k := i; k >= 0; k-- {
		inG := states[k].InG.Value()
		forG := states[k].ForG.Value()
		scores[k] = inG.Prod(incForgetProd).Max().Scalar().F64()
		if k > 0 {
			incForgetProd.ProdInPlace(forG)
		}
	}
	return scores
}
