// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lstmsc provides an implementation of LSTM enriched with a PolicyGradient
// to enable Dynamic Skip Connections.
package lstmsc

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	PolicyGradient *stack.Model
	Lambda         float64
	WIn            *nn.Param `type:"weights"`
	WInRec         *nn.Param `type:"weights"`
	BIn            *nn.Param `type:"biases"`
	WOut           *nn.Param `type:"weights"`
	WOutRec        *nn.Param `type:"weights"`
	BOut           *nn.Param `type:"biases"`
	WFor           *nn.Param `type:"weights"`
	WForRec        *nn.Param `type:"weights"`
	BFor           *nn.Param `type:"biases"`
	WCand          *nn.Param `type:"weights"`
	WCandRec       *nn.Param `type:"weights"`
	BCand          *nn.Param `type:"biases"`
}

// New returns a new model with parameters initialized to zeros.
// Lambda is the coefficient used in the equation λa + (1 − λ)b where 'a' is state[t-k] and 'b' is state[t-1].
func New(in, out, k int, lambda float64, intermediate int) *Model {
	var m Model
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
	return &m
}

func newGateParams(in, out int) (w, wRec, b *nn.Param) {
	w = nn.NewParam(mat.NewEmptyDense(out, in))
	wRec = nn.NewParam(mat.NewEmptyDense(out, out))
	b = nn.NewParam(mat.NewEmptyVecDense(out))
	return
}

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

type Processor struct {
	nn.BaseProcessor
	wIn            ag.Node
	wInRec         ag.Node
	bIn            ag.Node
	wOut           ag.Node
	wOutRec        ag.Node
	bOut           ag.Node
	wFor           ag.Node
	wForRec        ag.Node
	bFor           ag.Node
	wCand          ag.Node
	wCandRec       ag.Node
	bCand          ag.Node
	PolicyGradient *stack.Processor
	lambda         ag.Node
	negLambda      ag.Node
	States         []*State
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	g := ctx.Graph
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		States:         nil,
		wIn:            g.NewWrap(m.WIn),
		wInRec:         g.NewWrap(m.WInRec),
		bIn:            g.NewWrap(m.BIn),
		wOut:           g.NewWrap(m.WOut),
		wOutRec:        g.NewWrap(m.WOutRec),
		bOut:           g.NewWrap(m.BOut),
		wFor:           g.NewWrap(m.WFor),
		wForRec:        g.NewWrap(m.WForRec),
		bFor:           g.NewWrap(m.BFor),
		wCand:          g.NewWrap(m.WCand),
		wCandRec:       g.NewWrap(m.WCandRec),
		bCand:          g.NewWrap(m.BCand),
		lambda:         g.NewScalar(m.Lambda),
		negLambda:      g.NewScalar(1.0 - m.Lambda),
		PolicyGradient: m.PolicyGradient.NewProc(ctx).(*stack.Processor),
	}
}

func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("lstmsc: the initial state must be set before any input")
	}
	p.States = append(p.States, state)
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		s := p.forward(x)
		p.States = append(p.States, s)
		ys[i] = s.Y
	}
	return ys
}

func (p *Processor) LastState() *State {
	n := len(p.States)
	if n == 0 {
		return nil
	}
	return p.States[n-1]
}

func (p *Processor) PolicyGradientLogProbActions() []ag.Node {
	logPropActions := make([]ag.Node, len(p.States)-1)
	for i := range logPropActions {
		st := p.States[i+1] // skip the first state
		logPropActions[i] = p.Graph.Log(p.Graph.AtVec(st.Actions, st.SkipIndex))
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
func (p *Processor) forward(x ag.Node) (s *State) {
	g := p.Graph
	s = new(State)
	yPrev, cellPrev := p.prev()
	yPrevNew := yPrev
	cellPrevNew := cellPrev

	if yPrev != nil {
		s.Actions = p.PolicyGradient.Forward(g.NewWrapNoGrad(g.Concat(yPrev, x)))[0]
		s.SkipIndex = f64utils.ArgMax(s.Actions.Value().Data())
		if s.SkipIndex < len(p.States) {
			kState := p.States[len(p.States)-1-s.SkipIndex]
			yPrevNew = g.Add(g.ProdScalar(kState.Y, p.lambda), g.ProdScalar(yPrevNew, p.negLambda))
			cellPrevNew = g.Add(g.ProdScalar(kState.Cell, p.lambda), g.ProdScalar(cellPrevNew, p.negLambda))
		}
	}

	s.InG = g.Sigmoid(nn.Affine(g, p.bIn, p.wIn, x, p.wInRec, yPrevNew))
	s.OutG = g.Sigmoid(nn.Affine(g, p.bOut, p.wOut, x, p.wOutRec, yPrevNew))
	s.ForG = g.Sigmoid(nn.Affine(g, p.bFor, p.wFor, x, p.wForRec, yPrevNew))
	s.Cand = g.Tanh(nn.Affine(g, p.bCand, p.wCand, x, p.wCandRec, yPrevNew))
	if cellPrevNew != nil {
		s.Cell = g.Add(g.Prod(s.InG, s.Cand), g.Prod(s.ForG, cellPrevNew))
	} else {
		s.Cell = g.Prod(s.InG, s.Cand)
	}
	s.Y = g.Prod(s.OutG, g.Tanh(s.Cell))
	return
}

func (p *Processor) prev() (yPrev, cellPrev ag.Node) {
	s := p.LastState()
	if s != nil {
		yPrev = s.Y
		cellPrev = s.Cell
	}
	return
}
