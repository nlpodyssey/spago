// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lstm

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/initializers"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	UseRefinedGates bool

	// Input gate
	WIn    *nn.Param
	WInRec *nn.Param
	BIn    *nn.Param

	// Output gate
	WOut    *nn.Param
	WOutRec *nn.Param
	BOut    *nn.Param

	// Forget gate
	WFor    *nn.Param
	WForRec *nn.Param
	BFor    *nn.Param

	// Candiate gate
	WCand    *nn.Param
	WCandRec *nn.Param
	BCand    *nn.Param
}

// State represent a state of the LSTM recurrent network.
type State struct {
	InG  ag.DualValue
	OutG ag.DualValue
	ForG ag.DualValue
	Cand ag.DualValue
	Cell ag.DualValue
	Y    ag.DualValue
}

// Option allows to configure a new Model with your specific needs.
type Option func(*Model)

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](in, out int) *Model {
	return &Model{
		UseRefinedGates: false,

		// Input gate
		WIn:    nn.NewParam(mat.NewDense[T](mat.WithShape(out, in))),
		WInRec: nn.NewParam(mat.NewDense[T](mat.WithShape(out, out))),
		BIn:    nn.NewParam(mat.NewDense[T](mat.WithShape(out))),

		// Output gate
		WOut:    nn.NewParam(mat.NewDense[T](mat.WithShape(out, in))),
		WOutRec: nn.NewParam(mat.NewDense[T](mat.WithShape(out, out))),
		BOut:    nn.NewParam(mat.NewDense[T](mat.WithShape(out))),

		// Forget gate
		WFor:    nn.NewParam(mat.NewDense[T](mat.WithShape(out, in))),
		WForRec: nn.NewParam(mat.NewDense[T](mat.WithShape(out, out))),
		BFor:    nn.NewParam(mat.NewDense[T](mat.WithShape(out))),

		// Candiate gate
		WCand:    nn.NewParam(mat.NewDense[T](mat.WithShape(out, in))),
		WCandRec: nn.NewParam(mat.NewDense[T](mat.WithShape(out, out))),
		BCand:    nn.NewParam(mat.NewDense[T](mat.WithShape(out))),
	}
}

// WithRefinedGates sets whether to use refined gates.
// Refined Gate: A Simple and Effective Gating Mechanism for Recurrent Units
// (https://arxiv.org/pdf/2002.11338.pdf)
//
// Refined gates setting requires input size and output size be the same.
func (m *Model) WithRefinedGates(value bool) *Model {
	m.UseRefinedGates = value
	return m
}

// Init initializes the parameters using Xavier uniform randomization.
// It follows the LSTM bias hack setting the Forget gate to 1 (http://proceedings.mlr.press/v37/jozefowicz15.pdf).
func (m *Model) Init(rndGen *rand.LockedRand) *Model {
	nn.ForEachParam(m, func(param *nn.Param) {
		initializers.XavierUniform(param.Value(), 1, rndGen)
	})
	initializers.Constant(m.BFor.Value(), 1.0)
	return m
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.DualValue) []ag.DualValue {
	ys := make([]ag.DualValue, len(xs))
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
func (m *Model) Next(state *State, x ag.DualValue) (s *State) {
	s = new(State)

	var yPrev, cellPrev ag.DualValue = nil, nil
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
