// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gru

import (
	"brillion.io/spago/pkg/mat"
	"brillion.io/spago/pkg/ml/ag"
	"brillion.io/spago/pkg/ml/nn"
	"io"
	"log"
)

type Model struct {
	WPart    *nn.Param `type:"weights"`
	WPartRec *nn.Param `type:"weights"`
	BPart    *nn.Param `type:"biases"`
	WRes     *nn.Param `type:"weights"`
	WResRec  *nn.Param `type:"weights"`
	BRes     *nn.Param `type:"biases"`
	WCand    *nn.Param `type:"weights"`
	WCandRec *nn.Param `type:"weights"`
	BCand    *nn.Param `type:"biases"`
}

func New(in, out int) *Model {
	var m Model
	m.WPart, m.WPartRec, m.BPart = newGateParams(in, out)
	m.WRes, m.WResRec, m.BRes = newGateParams(in, out)
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
	R ag.Node
	P ag.Node
	C ag.Node
	Y ag.Node
}

type InitHidden struct {
	*State
}

type Processor struct {
	opt      []interface{}
	model    *Model
	g        *ag.Graph
	wPart    ag.Node
	wPartRec ag.Node
	bPart    ag.Node
	wRes     ag.Node
	wResRec  ag.Node
	bRes     ag.Node
	wCand    ag.Node
	wCandRec ag.Node
	bCand    ag.Node
	States   []*State
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:    m,
		States:   nil,
		opt:      opt,
		g:        g,
		wPart:    g.NewWrap(m.WPart),
		wPartRec: g.NewWrap(m.WPartRec),
		bPart:    g.NewWrap(m.BPart),
		wRes:     g.NewWrap(m.WRes),
		wResRec:  g.NewWrap(m.WResRec),
		bRes:     g.NewWrap(m.BRes),
		wCand:    g.NewWrap(m.WCand),
		wCandRec: g.NewWrap(m.WCandRec),
		bCand:    g.NewWrap(m.BCand),
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

func (p *Processor) Model() nn.Model {
	return p.model
}

func (p *Processor) Graph() *ag.Graph {
	return p.g
}

func (p *Processor) RequiresFullSeq() bool {
	return false
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

// r = sigmoid(wr (dot) x + br + wrRec (dot) yPrev)
// p = sigmoid(wp (dot) x + bp + wpRec (dot) yPrev)
// c = f(wc (dot) x + bc + wcRec (dot) (yPrev * r))
// y = p * c + (1 - p) * yPrev
func (p *Processor) forward(x ag.Node) (s *State) {
	s = new(State)
	yPrev := p.prev()
	s.R = p.g.Sigmoid(nn.Affine(p.g, p.bRes, p.wRes, x, p.wResRec, yPrev))
	s.P = p.g.Sigmoid(nn.Affine(p.g, p.bPart, p.wPart, x, p.wPartRec, yPrev))
	s.C = p.g.Tanh(nn.Affine(p.g, p.bCand, p.wCand, x, p.wCandRec, tryProd(p.g, yPrev, s.R)))
	s.Y = p.g.Prod(s.P, s.C)
	if yPrev != nil {
		s.Y = p.g.Add(s.Y, p.g.Prod(p.g.ReverseSub(s.P, p.g.NewScalar(1.0)), yPrev))
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

// tryProd returns the product if 'a' il not nil, otherwise nil
func tryProd(g *ag.Graph, a, b ag.Node) ag.Node {
	if a != nil {
		return g.Prod(a, b)
	}
	return nil
}
