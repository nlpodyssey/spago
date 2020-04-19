// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lstmsc

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/perceptron"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"io"
	"log"
)

var _ nn.Model = &Model{}

// LSTM enriched with a PolicyGradient to enable Dynamic Skip Connections.
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

// New returns a new model.
// lambda is the coefficient used in the equation λa + (1 − λ)b where 'a' is state[t-k] and 'b' is state[t-1].
func New(in, out, k int, lambda float64, intermediate int) *Model {
	var m Model
	m.PolicyGradient = stack.New(
		perceptron.New(in+out, intermediate, ag.Tanh),
		perceptron.New(intermediate, k, ag.Softmax),
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

func (m *Model) ForEachParam(callback func(param *nn.Param)) {
	nn.ForEachParam(m, callback)
}

func (m *Model) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

func (m *Model) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
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

type InitHidden struct {
	*State
}

type GateConcurrency struct {
	Value bool
}

var _ nn.Processor = &Processor{}

type Processor struct {
	opt             []interface{}
	model           *Model
	mode            nn.ProcessingMode
	g               *ag.Graph
	wIn             ag.Node
	wInRec          ag.Node
	bIn             ag.Node
	wOut            ag.Node
	wOutRec         ag.Node
	bOut            ag.Node
	wFor            ag.Node
	wForRec         ag.Node
	bFor            ag.Node
	wCand           ag.Node
	wCandRec        ag.Node
	bCand           ag.Node
	PolicyGradient  *stack.Processor
	lambda          ag.Node
	negLambda       ag.Node
	States          []*State
	GateConcurrency bool
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:           m,
		mode:            nn.Training,
		States:          nil,
		GateConcurrency: false,
		opt:             opt,
		g:               g,
		wIn:             g.NewWrap(m.WIn),
		wInRec:          g.NewWrap(m.WInRec),
		bIn:             g.NewWrap(m.BIn),
		wOut:            g.NewWrap(m.WOut),
		wOutRec:         g.NewWrap(m.WOutRec),
		bOut:            g.NewWrap(m.BOut),
		wFor:            g.NewWrap(m.WFor),
		wForRec:         g.NewWrap(m.WForRec),
		bFor:            g.NewWrap(m.BFor),
		wCand:           g.NewWrap(m.WCand),
		wCandRec:        g.NewWrap(m.WCandRec),
		bCand:           g.NewWrap(m.BCand),
		lambda:          g.NewScalar(m.Lambda),
		negLambda:       g.NewScalar(1.0 - m.Lambda),
		PolicyGradient:  m.PolicyGradient.NewProc(g).(*stack.Processor),
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	for _, t := range opt {
		switch t := t.(type) {
		case InitHidden:
			p.States = append(p.States, t.State)
		case GateConcurrency:
			p.GateConcurrency = t.Value
		default:
			log.Fatal("lstmrl: invalid init option")
		}
	}
}

func (p *Processor) Model() nn.Model         { return p.model }
func (p *Processor) Graph() *ag.Graph        { return p.g }
func (p *Processor) RequiresFullSeq() bool   { return false }
func (p *Processor) Mode() nn.ProcessingMode { return p.mode }

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.mode = mode
	p.PolicyGradient.SetMode(mode)
}

func (p *Processor) Reset() {
	p.States = nil
	p.init(p.opt)
}

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
		logPropActions[i] = p.g.Log(p.g.AtVec(st.Actions, st.SkipIndex))
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
	s = new(State)
	yPrev, cellPrev := p.prev()

	yPrevNew := yPrev
	cellPrevNew := cellPrev
	if yPrev != nil {
		s.Actions = p.PolicyGradient.Forward(p.g.NewWrapNoGrad(p.g.Concat(yPrev, x)))[0]
		s.SkipIndex = f64utils.ArgMax(s.Actions.Value().Data())
		if s.SkipIndex < len(p.States) {
			kState := p.States[len(p.States)-1-s.SkipIndex]
			yPrevNew = p.g.Add(p.g.ProdScalar(kState.Y, p.lambda), p.g.ProdScalar(yPrevNew, p.negLambda))
			cellPrevNew = p.g.Add(p.g.ProdScalar(kState.Cell, p.lambda), p.g.ProdScalar(cellPrevNew, p.negLambda))
		}
	}

	s.InG = p.g.Sigmoid(nn.Affine(p.g, p.bIn, p.wIn, x, p.wInRec, yPrevNew))
	s.OutG = p.g.Sigmoid(nn.Affine(p.g, p.bOut, p.wOut, x, p.wOutRec, yPrevNew))
	s.ForG = p.g.Sigmoid(nn.Affine(p.g, p.bFor, p.wFor, x, p.wForRec, yPrevNew))
	s.Cand = p.g.Tanh(nn.Affine(p.g, p.bCand, p.wCand, x, p.wCandRec, yPrevNew))
	if cellPrevNew != nil {
		s.Cell = p.g.Add(p.g.Prod(s.InG, s.Cand), p.g.Prod(s.ForG, cellPrevNew))
	} else {
		s.Cell = p.g.Prod(s.InG, s.Cand)
	}
	s.Y = p.g.Prod(s.OutG, p.g.Tanh(s.Cell))
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
