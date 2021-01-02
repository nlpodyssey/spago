// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lstmsc provides an implementation of LSTM enriched with a PolicyGradient
// to enable Dynamic Skip Connections.
package lstmsc

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"log"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	PolicyGradient *stack.Model
	Lambda         mat.Float
	WIn            nn.Param `spago:"type:weights"`
	WInRec         nn.Param `spago:"type:weights"`
	BIn            nn.Param `spago:"type:biases"`
	WOut           nn.Param `spago:"type:weights"`
	WOutRec        nn.Param `spago:"type:weights"`
	BOut           nn.Param `spago:"type:biases"`
	WFor           nn.Param `spago:"type:weights"`
	WForRec        nn.Param `spago:"type:weights"`
	BFor           nn.Param `spago:"type:biases"`
	WCand          nn.Param `spago:"type:weights"`
	WCandRec       nn.Param `spago:"type:weights"`
	BCand          nn.Param `spago:"type:biases"`
	States         []*State `spago:"scope:processor"`
}

// New returns a new model with parameters initialized to zeros.
// Lambda is the coefficient used in the equation λa + (1 − λ)b where 'a' is state[t-k] and 'b' is state[t-1].
func New(in, out, k int, lambda mat.Float, intermediate int) *Model {
	m := &Model{}
	m.PolicyGradient = stack.New(
		linear.New(in+out, intermediate),
		activation.New(ag.OpTanh),
		linear.New(intermediate, k),
		activation.New(ag.OpSoftmax),
	)
	m.Lambda = lambda
	m.WIn, m.WInRec, m.BIn = newGateParams(in, out)
	m.WOut, m.WOutRec, m.BOut = newGateParams(in, out)
	m.WFor, m.WForRec, m.BFor = newGateParams(in, out)
	m.WCand, m.WCandRec, m.BCand = newGateParams(in, out)
	return m
}

// State represent a state of the LSTM with Dynamic Skip Connections recurrent network.
type State struct {
	InG       ag.Node
	OutG      ag.Node
	ForG      ag.Node
	Cand      ag.Node
	Cell      ag.Node
	Y         ag.Node
	Actions   ag.Node
	SkipIndex int
}

func newGateParams(in, out int) (w, wRec, b nn.Param) {
	w = nn.NewParam(mat.NewEmptyDense(out, in))
	wRec = nn.NewParam(mat.NewEmptyDense(out, out))
	b = nn.NewParam(mat.NewEmptyVecDense(out))
	return
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model) SetInitialState(state *State) {
	if len(m.States) > 0 {
		log.Fatal("lstmsc: the initial state must be set before any input")
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

// PolicyGradientLogProbActions returns the log probabilities for each action
// estimated by the policy gradient.
func (m *Model) PolicyGradientLogProbActions() []ag.Node {
	g := m.Graph()
	logPropActions := make([]ag.Node, len(m.States)-1)
	for i := range logPropActions {
		st := m.States[i+1] // skip the first state
		logPropActions[i] = g.Log(g.AtVec(st.Actions, st.SkipIndex))
	}
	return logPropActions
}

// forward computes the results with the following equations:
// inG = sigmoid(wIn (dot) x + bIn + wInRec (dot) yPrev)
// outG = sigmoid(wOut (dot) x + bOut + wOutRec (dot) yPrev)
// forG = sigmoid(wFor (dot) x + bFor + wForRec (dot) yPrev)
// cand = f(wCand (dot) x + bC + wCandRec (dot) yPrev)
// cell = inG * cand + forG * cellPrev
// y = outG * f(cell)
func (m *Model) forward(x ag.Node) (s *State) {
	g := m.Graph()
	s = new(State)
	yPrev, cellPrev := m.prev()
	yPrevNew := yPrev
	cellPrevNew := cellPrev
	lambda := g.NewScalar(m.Lambda)
	negLambda := g.NewScalar(1.0 - m.Lambda)

	if yPrev != nil {
		s.Actions = nn.ToNode(m.PolicyGradient.Forward(g.NewWrapNoGrad(g.Concat(yPrev, x))))
		s.SkipIndex = floatutils.ArgMax(s.Actions.Value().Data())
		if s.SkipIndex < len(m.States) {
			kState := m.States[len(m.States)-1-s.SkipIndex]
			yPrevNew = g.Add(g.ProdScalar(kState.Y, lambda), g.ProdScalar(yPrevNew, negLambda))
			cellPrevNew = g.Add(g.ProdScalar(kState.Cell, lambda), g.ProdScalar(cellPrevNew, negLambda))
		}
	}

	s.InG = g.Sigmoid(nn.Affine(g, m.BIn, m.WIn, x, m.WInRec, yPrevNew))
	s.OutG = g.Sigmoid(nn.Affine(g, m.BOut, m.WOut, x, m.WOutRec, yPrevNew))
	s.ForG = g.Sigmoid(nn.Affine(g, m.BFor, m.WFor, x, m.WForRec, yPrevNew))
	s.Cand = g.Tanh(nn.Affine(g, m.BCand, m.WCand, x, m.WCandRec, yPrevNew))
	if cellPrevNew != nil {
		s.Cell = g.Add(g.Prod(s.InG, s.Cand), g.Prod(s.ForG, cellPrevNew))
	} else {
		s.Cell = g.Prod(s.InG, s.Cand)
	}
	s.Y = g.Prod(s.OutG, g.Tanh(s.Cell))
	return
}

func (m *Model) prev() (yPrev, cellPrev ag.Node) {
	s := m.LastState()
	if s != nil {
		yPrev = s.Y
		cellPrev = s.Cell
	}
	return
}
