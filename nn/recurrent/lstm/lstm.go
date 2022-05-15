// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lstm

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	UseRefinedGates bool

	// Input gate
	WIn    nn.Param `spago:"type:weights"`
	WInRec nn.Param `spago:"type:weights"`
	BIn    nn.Param `spago:"type:biases"`

	// Output gate
	WOut    nn.Param `spago:"type:weights"`
	WOutRec nn.Param `spago:"type:weights"`
	BOut    nn.Param `spago:"type:biases"`

	// Forget gate
	WFor    nn.Param `spago:"type:weights"`
	WForRec nn.Param `spago:"type:weights"`
	BFor    nn.Param `spago:"type:biases"`

	// Candiate gate
	WCand    nn.Param `spago:"type:weights"`
	WCandRec nn.Param `spago:"type:weights"`
	BCand    nn.Param `spago:"type:biases"`
}

// State represent a state of the LSTM recurrent network.
type State struct {
	InG  ag.Node
	OutG ag.Node
	ForG ag.Node
	Cand ag.Node
	Cell ag.Node
	Y    ag.Node
}

// Option allows to configure a new Model with your specific needs.
type Option func(*Model)

func init() {
	gob.Register(&Model{})
}

// SetRefinedGates sets whether to use refined gates.
// Refined Gate: A Simple and Effective Gating Mechanism for Recurrent Units
// (https://arxiv.org/pdf/2002.11338.pdf)
// TODO: panic input size and output size are different
func SetRefinedGates(value bool) Option {
	return func(m *Model) {
		m.UseRefinedGates = value
	}
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int, options ...Option) *Model {
	m := &Model{
		UseRefinedGates: false,

		// Input gate
		WIn:    nn.NewParam(mat.NewEmptyDense[T](out, in)),
		WInRec: nn.NewParam(mat.NewEmptyDense[T](out, out)),
		BIn:    nn.NewParam(mat.NewEmptyVecDense[T](out)),

		// Output gate
		WOut:    nn.NewParam(mat.NewEmptyDense[T](out, in)),
		WOutRec: nn.NewParam(mat.NewEmptyDense[T](out, out)),
		BOut:    nn.NewParam(mat.NewEmptyVecDense[T](out)),

		// Forget gate
		WFor:    nn.NewParam(mat.NewEmptyDense[T](out, in)),
		WForRec: nn.NewParam(mat.NewEmptyDense[T](out, out)),
		BFor:    nn.NewParam(mat.NewEmptyVecDense[T](out)),

		// Candiate gate
		WCand:    nn.NewParam(mat.NewEmptyDense[T](out, in)),
		WCandRec: nn.NewParam(mat.NewEmptyDense[T](out, out)),
		BCand:    nn.NewParam(mat.NewEmptyVecDense[T](out)),
	}

	for _, option := range options {
		option(m)
	}
	return m
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
// It computes the results with the following equations:
// inG = sigmoid(wIn (dot) x + bIn + wInRec (dot) yPrev)
// outG = sigmoid(wOut (dot) x + bOut + wOutRec (dot) yPrev)
// forG = sigmoid(wFor (dot) x + bFor + wForRec (dot) yPrev)
// cand = f(wCand (dot) x + bC + wCandRec (dot) yPrev)
// cell = inG * cand + forG * cellPrev
// y = outG * f(cell)
func (m *Model) Next(state *State, x ag.Node) (s *State) {
	s = new(State)

	var yPrev, cellPrev ag.Node = nil, nil
	if state != nil {
		yPrev, cellPrev = state.Y, state.Cell
	}

	s.InG = ag.Sigmoid(ag.Affine(m.BIn, m.WIn, x, m.WInRec, yPrev))
	s.OutG = ag.Sigmoid(ag.Affine(m.BOut, m.WOut, x, m.WOutRec, yPrev))
	s.ForG = ag.Sigmoid(ag.Affine(m.BFor, m.WFor, x, m.WForRec, yPrev))
	s.Cand = ag.Tanh(ag.Affine(m.BCand, m.WCand, x, m.WCandRec, yPrev))

	if m.UseRefinedGates {
		s.InG = ag.Prod(s.InG, x)
		s.OutG = ag.Prod(s.OutG, x)
	}

	if cellPrev != nil {
		s.Cell = ag.Add(ag.Prod(s.InG, s.Cand), ag.Prod(s.ForG, cellPrev))
	} else {
		s.Cell = ag.Prod(s.InG, s.Cand)
	}
	s.Y = ag.Prod(s.OutG, ag.Tanh(s.Cell))
	return
}
