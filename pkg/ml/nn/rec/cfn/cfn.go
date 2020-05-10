// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfn

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
	WIn     *nn.Param `type:"weights"`
	WInRec  *nn.Param `type:"weights"`
	BIn     *nn.Param `type:"biases"`
	WFor    *nn.Param `type:"weights"`
	WForRec *nn.Param `type:"weights"`
	BFor    *nn.Param `type:"biases"`
	WCand   *nn.Param `type:"weights"`
}

func New(in, out int) *Model {
	var m Model
	m.WIn, m.WInRec, m.BIn = newGateParams(in, out)
	m.WFor, m.WForRec, m.BFor = newGateParams(in, out)
	m.WCand = nn.NewParam(mat.NewEmptyDense(out, in))
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
	Y    ag.Node
}

type InitHidden struct {
	*State
}

type Processor struct {
	opt     []interface{}
	model   *Model
	mode    nn.ProcessingMode
	g       *ag.Graph
	wIn     ag.Node
	wInRec  ag.Node
	bIn     ag.Node
	wFor    ag.Node
	wForRec ag.Node
	bFor    ag.Node
	wCand   ag.Node
	States  []*State
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:   m,
		mode:    nn.Training,
		States:  nil,
		opt:     opt,
		g:       g,
		wIn:     g.NewWrap(m.WIn),
		wInRec:  g.NewWrap(m.WInRec),
		bIn:     g.NewWrap(m.BIn),
		wFor:    g.NewWrap(m.WFor),
		wForRec: g.NewWrap(m.WForRec),
		bFor:    g.NewWrap(m.BFor),
		wCand:   g.NewWrap(m.WCand),
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	for _, t := range opt {
		switch t := t.(type) {
		case InitHidden:
			p.States = append(p.States, t.State)
		default:
			log.Fatal("srn: invalid init option")
		}
	}
}

func (p *Processor) Model() nn.Model                { return p.model }
func (p *Processor) Graph() *ag.Graph               { return p.g }
func (p *Processor) RequiresFullSeq() bool          { return false }
func (p *Processor) Mode() nn.ProcessingMode        { return p.mode }
func (p *Processor) SetMode(mode nn.ProcessingMode) { p.mode = mode }

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
// c = f(wc (dot) x)
// y = inG * c + f(yPrev) * forG
func (p *Processor) forward(x ag.Node) (s *State) {
	s = new(State)
	yPrev := p.prev()
	s.InG = p.g.Sigmoid(nn.Affine(p.g, p.bIn, p.wIn, x, p.wInRec, yPrev))
	s.ForG = p.g.Sigmoid(nn.Affine(p.g, p.bFor, p.wFor, x, p.wForRec, yPrev))
	s.Cand = p.g.Tanh(p.g.Mul(p.wCand, x))
	s.Y = p.g.Prod(s.InG, s.Cand)
	if yPrev != nil {
		s.Y = p.g.Add(s.Y, p.g.Prod(p.g.Tanh(yPrev), s.ForG))
	}
	return
}

func (p *Processor) prev() (yPrev ag.Node) {
	s := p.LastState()
	if s != nil {
		yPrev = s.Y
	}
	return
}
