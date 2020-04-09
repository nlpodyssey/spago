// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ltm

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"io"
	"log"
)

var _ nn.Model = &Model{}

type Model struct {
	W1    *nn.Param `type:"weights"`
	W2    *nn.Param `type:"weights"`
	W3    *nn.Param `type:"weights"`
	WCell *nn.Param `type:"weights"`
}

func New(in int) *Model {
	return &Model{
		W1:    nn.NewParam(mat.NewEmptyDense(in, in)),
		W2:    nn.NewParam(mat.NewEmptyDense(in, in)),
		W3:    nn.NewParam(mat.NewEmptyDense(in, in)),
		WCell: nn.NewParam(mat.NewEmptyDense(in, in)),
	}
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
	L1   ag.Node
	L2   ag.Node
	L3   ag.Node
	Cand ag.Node
	Cell ag.Node
	Y    ag.Node
}

type InitHidden struct {
	*State
}

var _ nn.Processor = &Processor{}

type Processor struct {
	opt    []interface{}
	model  *Model
	mode   nn.ProcessingMode
	g      *ag.Graph
	w1     ag.Node
	w2     ag.Node
	w3     ag.Node
	wCell  ag.Node
	States []*State
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:  m,
		mode:   nn.Training,
		States: nil,
		opt:    opt,
		g:      g,
		w1:     g.NewWrap(m.W1),
		w2:     g.NewWrap(m.W2),
		w3:     g.NewWrap(m.W3),
		wCell:  g.NewWrap(m.WCell),
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

// l1 = sigmoid(w1 (dot) (x + yPrev))
// l2 = sigmoid(w2 (dot) (x + yPrev))
// l3 = sigmoid(w3 (dot) (x + yPrev))
// c = l1 * l2 + cellPrev
// cell = sigmoid(c (dot) wCell + bCell)
// y = cell * l3
func (p *Processor) forward(x ag.Node) (s *State) {
	s = new(State)
	yPrev, cellPrev := p.prev()
	h := x
	if yPrev != nil {
		h = p.g.Add(h, yPrev)
	}
	s.L1 = p.g.Sigmoid(nn.Linear(p.g, p.w1, h))
	s.L2 = p.g.Sigmoid(nn.Linear(p.g, p.w2, h))
	s.L3 = p.g.Sigmoid(nn.Linear(p.g, p.w3, h))
	s.Cand = p.g.Prod(s.L1, s.L2)
	if cellPrev != nil {
		s.Cand = p.g.Add(s.Cand, cellPrev)
	}
	s.Cell = p.g.Sigmoid(nn.Linear(p.g, p.wCell, s.Cand))
	s.Y = p.g.Prod(s.Cell, s.L3)
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
