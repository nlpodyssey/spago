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

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	UseRefinedGates bool

	// Input gate
	WIn    nn.Param[T] `spago:"type:weights"`
	WInRec nn.Param[T] `spago:"type:weights"`
	BIn    nn.Param[T] `spago:"type:biases"`

	// Output gate
	WOut    nn.Param[T] `spago:"type:weights"`
	WOutRec nn.Param[T] `spago:"type:weights"`
	BOut    nn.Param[T] `spago:"type:biases"`

	// Forget gate
	WFor    nn.Param[T] `spago:"type:weights"`
	WForRec nn.Param[T] `spago:"type:weights"`
	BFor    nn.Param[T] `spago:"type:biases"`

	// Candiate gate
	WCand    nn.Param[T] `spago:"type:weights"`
	WCandRec nn.Param[T] `spago:"type:weights"`
	BCand    nn.Param[T] `spago:"type:biases"`
}

// State represent a state of the LSTM recurrent network.
type State[T mat.DType] struct {
	InG  ag.Node[T]
	OutG ag.Node[T]
	ForG ag.Node[T]
	Cand ag.Node[T]
	Cell ag.Node[T]
	Y    ag.Node[T]
}

// Option allows to configure a new Model with your specific needs.
type Option[T mat.DType] func(*Model[T])

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// SetRefinedGates sets whether to use refined gates.
// Refined Gate: A Simple and Effective Gating Mechanism for Recurrent Units
// (https://arxiv.org/pdf/2002.11338.pdf)
// TODO: panic input size and output size are different
func SetRefinedGates[T mat.DType](value bool) Option[T] {
	return func(m *Model[T]) {
		m.UseRefinedGates = value
	}
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](in, out int, options ...Option[T]) *Model[T] {
	m := &Model[T]{
		UseRefinedGates: false,

		// Input gate
		WIn:    nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		WInRec: nn.NewParam[T](mat.NewEmptyDense[T](out, out)),
		BIn:    nn.NewParam[T](mat.NewEmptyVecDense[T](out)),

		// Output gate
		WOut:    nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		WOutRec: nn.NewParam[T](mat.NewEmptyDense[T](out, out)),
		BOut:    nn.NewParam[T](mat.NewEmptyVecDense[T](out)),

		// Forget gate
		WFor:    nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		WForRec: nn.NewParam[T](mat.NewEmptyDense[T](out, out)),
		BFor:    nn.NewParam[T](mat.NewEmptyVecDense[T](out)),

		// Candiate gate
		WCand:    nn.NewParam[T](mat.NewEmptyDense[T](out, in)),
		WCandRec: nn.NewParam[T](mat.NewEmptyDense[T](out, out)),
		BCand:    nn.NewParam[T](mat.NewEmptyVecDense[T](out)),
	}

	for _, option := range options {
		option(m)
	}
	return m
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
// It computes the results with the following equations:
// inG = sigmoid(wIn (dot) x + bIn + wInRec (dot) yPrev)
// outG = sigmoid(wOut (dot) x + bOut + wOutRec (dot) yPrev)
// forG = sigmoid(wFor (dot) x + bFor + wForRec (dot) yPrev)
// cand = f(wCand (dot) x + bC + wCandRec (dot) yPrev)
// cell = inG * cand + forG * cellPrev
// y = outG * f(cell)
func (m *Model[T]) Next(state *State[T], x ag.Node[T]) (s *State[T]) {
	s = new(State[T])

	var yPrev, cellPrev ag.Node[T] = nil, nil
	if state != nil {
		yPrev, cellPrev = state.Y, state.Cell
	}

	s.InG = ag.Sigmoid(ag.Affine[T](m.BIn, m.WIn, x, m.WInRec, yPrev))
	s.OutG = ag.Sigmoid(ag.Affine[T](m.BOut, m.WOut, x, m.WOutRec, yPrev))
	s.ForG = ag.Sigmoid(ag.Affine[T](m.BFor, m.WFor, x, m.WForRec, yPrev))
	s.Cand = ag.Tanh(ag.Affine[T](m.BCand, m.WCand, x, m.WCandRec, yPrev))

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
