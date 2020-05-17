// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fsmn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// This is variant of the Feedforward Sequential Memory Networks (https://arxiv.org/pdf/1512.08301.pdf) where the
// neurons in the same hidden layer are independent of each other and they are connected across layers as in the IndRNN.
type Model struct {
	W     *nn.Param   `type:"weights"`
	WRec  *nn.Param   `type:"weights"`
	WS    []*nn.Param `type:"weights"` // coefficient vectors for scaling
	B     *nn.Param   `type:"biases"`
	order int
}

func New(in, out, order int) *Model {
	WS := make([]*nn.Param, order, order)
	for i := 0; i < order; i++ {
		WS[i] = nn.NewParam(mat.NewEmptyVecDense(out))
	}
	return &Model{
		W:     nn.NewParam(mat.NewEmptyDense(out, in)),
		WRec:  nn.NewParam(mat.NewEmptyVecDense(out)),
		WS:    WS,
		B:     nn.NewParam(mat.NewEmptyVecDense(out)),
		order: order,
	}
}

type State struct {
	Y ag.Node
}

type InitHidden struct {
	*State
}

type Processor struct {
	nn.BaseProcessor
	order  int
	w      ag.Node
	wRec   ag.Node
	wS     []ag.Node
	b      ag.Node
	States []*State
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	wS := make([]ag.Node, len(m.WS))
	for i, p := range m.WS {
		wS[i] = g.NewWrap(p)
	}
	p := &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		order:  m.order,
		States: nil,
		w:      g.NewWrap(m.W),
		wRec:   g.NewWrap(m.WRec),
		wS:     wS,
		b:      g.NewWrap(m.B),
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
			log.Fatal("fsmn: invalid init option")
		}
	}
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

func (p *Processor) forward(x ag.Node) (s *State) {
	g := p.Graph
	s = new(State)
	h := nn.Affine(g, p.b, p.w, x)
	if len(p.States) > 0 {
		h = g.Add(h, g.Prod(p.wRec, p.feedback()))
	}
	s.Y = g.ReLU(h)
	return
}

func (p *Processor) feedback() ag.Node {
	g := p.Graph
	var y ag.Node
	n := len(p.States)
	min := utils.MinInt(p.order, n)
	for i := 0; i < min; i++ {
		scaled := g.Prod(p.wS[i], g.NewWrapNoGrad(p.States[n-1-i].Y))
		if y == nil {
			y = scaled
		} else {
			y = g.Add(y, scaled)
		}
	}
	return y
}
