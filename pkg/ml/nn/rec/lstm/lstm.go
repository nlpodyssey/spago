// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lstm

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

type Model struct {
	WIn      *nn.Param `type:"weights"`
	WInRec   *nn.Param `type:"weights"`
	BIn      *nn.Param `type:"biases"`
	WOut     *nn.Param `type:"weights"`
	WOutRec  *nn.Param `type:"weights"`
	BOut     *nn.Param `type:"biases"`
	WFor     *nn.Param `type:"weights"`
	WForRec  *nn.Param `type:"weights"`
	BFor     *nn.Param `type:"biases"`
	WCand    *nn.Param `type:"weights"`
	WCandRec *nn.Param `type:"weights"`
	BCand    *nn.Param `type:"biases"`
}

func New(in, out int) *Model {
	var m Model
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
	InG  ag.Node
	OutG ag.Node
	ForG ag.Node
	Cand ag.Node
	Cell ag.Node
	Y    ag.Node
}

type Processor struct {
	nn.BaseProcessor
	wIn      ag.Node
	wInRec   ag.Node
	bIn      ag.Node
	wOut     ag.Node
	wOutRec  ag.Node
	bOut     ag.Node
	wFor     ag.Node
	wForRec  ag.Node
	bFor     ag.Node
	wCand    ag.Node
	wCandRec ag.Node
	bCand    ag.Node
	States   []*State
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		States:   nil,
		wIn:      g.NewWrap(m.WIn),
		wInRec:   g.NewWrap(m.WInRec),
		bIn:      g.NewWrap(m.BIn),
		wOut:     g.NewWrap(m.WOut),
		wOutRec:  g.NewWrap(m.WOutRec),
		bOut:     g.NewWrap(m.BOut),
		wFor:     g.NewWrap(m.WFor),
		wForRec:  g.NewWrap(m.WForRec),
		bFor:     g.NewWrap(m.BFor),
		wCand:    g.NewWrap(m.WCand),
		wCandRec: g.NewWrap(m.WCandRec),
		bCand:    g.NewWrap(m.BCand),
	}
}

func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("lstm: the initial state must be set before any input")
	}
	p.States = append(p.States, state)
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
	s.InG = g.Sigmoid(nn.Affine(g, p.bIn, p.wIn, x, p.wInRec, yPrev))
	s.OutG = g.Sigmoid(nn.Affine(g, p.bOut, p.wOut, x, p.wOutRec, yPrev))
	s.ForG = g.Sigmoid(nn.Affine(g, p.bFor, p.wFor, x, p.wForRec, yPrev))
	s.Cand = g.Tanh(nn.Affine(g, p.bCand, p.wCand, x, p.wCandRec, yPrev))
	if cellPrev != nil {
		s.Cell = g.Add(g.Prod(s.InG, s.Cand), g.Prod(s.ForG, cellPrev))
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
