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

// Model implements a variant of the Feedforward Sequential Memory Networks
// (https://arxiv.org/pdf/1512.08301.pdf) where the neurons in the same hidden layer
// are independent of each other and they are connected across layers as in the IndRNN.
type Model struct {
	W     *nn.Param   `type:"weights"`
	WRec  *nn.Param   `type:"weights"`
	WS    []*nn.Param `type:"weights"` // coefficient vectors for scaling
	B     *nn.Param   `type:"biases"`
	order int
}

// New returns a new model with parameters initialized to zeros.
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

type Processor struct {
	nn.BaseProcessor
	order  int
	w      ag.Node
	wRec   ag.Node
	wS     []ag.Node
	b      ag.Node
	States []*State
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	g := ctx.Graph
	wS := make([]ag.Node, len(m.WS))
	for i, p := range m.WS {
		wS[i] = g.NewWrap(p)
	}
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		order:  m.order,
		States: nil,
		w:      g.NewWrap(m.W),
		wRec:   g.NewWrap(m.WRec),
		wS:     wS,
		b:      g.NewWrap(m.B),
	}
}

func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("fsmn: the initial state must be set before any input")
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
