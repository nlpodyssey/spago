// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ran

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	WIn     *nn.Param `type:"weights"`
	WInRec  *nn.Param `type:"weights"`
	BIn     *nn.Param `type:"biases"`
	WFor    *nn.Param `type:"weights"`
	WForRec *nn.Param `type:"weights"`
	BFor    *nn.Param `type:"biases"`
	WCand   *nn.Param `type:"weights"`
	BCand   *nn.Param `type:"biases"`
}

// New returns a new model with parameters initialized to zeros.
func New(in, out int) *Model {
	var m Model
	m.WIn, m.WInRec, m.BIn = newGateParams(in, out)
	m.WFor, m.WForRec, m.BFor = newGateParams(in, out)
	m.WCand = nn.NewParam(mat.NewEmptyDense(out, in))
	m.BCand = nn.NewParam(mat.NewEmptyVecDense(out))
	return &m
}

func newGateParams(in, out int) (w, wRec, b *nn.Param) {
	w = nn.NewParam(mat.NewEmptyDense(out, in))
	wRec = nn.NewParam(mat.NewEmptyDense(out, out))
	b = nn.NewParam(mat.NewEmptyVecDense(out))
	return
}

type State struct {
	InG  ag.Node
	ForG ag.Node
	Cand ag.Node
	C    ag.Node
	Y    ag.Node
}

type Processor struct {
	nn.BaseProcessor
	wIn     ag.Node
	wInRec  ag.Node
	bIn     ag.Node
	wFor    ag.Node
	wForRec ag.Node
	bFor    ag.Node
	wCand   ag.Node
	bCand   ag.Node
	States  []*State
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
		wIn:     g.NewWrap(m.WIn),
		wInRec:  g.NewWrap(m.WInRec),
		bIn:     g.NewWrap(m.BIn),
		wFor:    g.NewWrap(m.WFor),
		wForRec: g.NewWrap(m.WForRec),
		bFor:    g.NewWrap(m.BFor),
		wCand:   g.NewWrap(m.WCand),
		bCand:   g.NewWrap(m.BCand),
		States:  nil,
	}
}

func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("ran: the initial state must be set before any input")
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

// inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
// forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
// cand = wc (dot) x + bc
// c = inG * c + forG * cPrev
// y = f(c)
func (p *Processor) forward(x ag.Node) (s *State) {
	g := p.Graph
	s = new(State)
	yPrev, cPrev := p.prev()
	s.InG = g.Sigmoid(nn.Affine(g, p.bIn, p.wIn, x, p.wInRec, yPrev))
	s.ForG = g.Sigmoid(nn.Affine(g, p.bFor, p.wFor, x, p.wForRec, yPrev))
	s.Cand = nn.Affine(g, p.bCand, p.wCand, x)
	s.C = g.Prod(s.InG, s.Cand)
	if cPrev != nil {
		s.C = g.Add(s.C, g.Prod(s.ForG, cPrev))
	}
	s.Y = g.Tanh(s.C)
	return
}

func (p *Processor) prev() (yPrev, cPrev ag.Node) {
	s := p.LastState()
	if s != nil {
		yPrev = s.Y
		cPrev = s.Y
	}
	return
}

func (p *Processor) Importance() [][]float64 {
	importance := make([][]float64, len(p.States))
	for i := range importance {
		importance[i] = p.scores(i)
	}
	return importance
}

// importance computes the importance score of the previous states respect to the i-state.
// The output contains the importance score for each k-previous states.
func (p *Processor) scores(i int) []float64 {
	states := p.States
	scores := make([]float64, len(states))
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
