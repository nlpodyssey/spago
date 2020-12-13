// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package horn provides an implementation of Higher Order Recurrent Neural Networks (HORN).
package horn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
	"math"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	W    *nn.Param   `type:"weights"`
	WRec []*nn.Param `type:"weights"`
	B    *nn.Param   `type:"biases"`
}

// New returns a new model with parameters initialized to zeros.
func New(in, out, order int) *Model {
	wRec := make([]*nn.Param, order, order)
	for i := 0; i < order; i++ {
		wRec[i] = nn.NewParam(mat.NewEmptyDense(out, out))
	}
	return &Model{
		W:    nn.NewParam(mat.NewEmptyDense(out, in)),
		WRec: wRec,
		B:    nn.NewParam(mat.NewEmptyVecDense(out)),
	}
}

type State struct {
	Y ag.Node
}

type Processor struct {
	nn.BaseProcessor
	w      ag.Node
	wRec   []ag.Node
	b      ag.Node
	States []*State
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	g := ctx.Graph
	wRec := make([]ag.Node, len(m.WRec))
	for i, p := range m.WRec {
		wRec[i] = g.NewWrap(p)
	}
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		States: nil,
		w:      g.NewWrap(m.W),
		wRec:   wRec,
		b:      g.NewWrap(m.B),
	}
}

func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("horn: the initial state must be set before any input")
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
	s = new(State)
	h := nn.Affine(p.Graph, append([]ag.Node{p.b, p.w, x}, p.feedback()...)...)
	s.Y = p.Graph.Tanh(h)
	return
}

func (p *Processor) feedback() []ag.Node {
	var ys []ag.Node
	n := len(p.States)
	for i := 0; i < utils.MinInt(len(p.wRec), n); i++ {
		alpha := p.Graph.NewScalar(math.Pow(0.6, float64(i+1)))
		ys = append(ys, p.wRec[i], p.Graph.ProdScalar(p.States[n-1-i].Y, alpha))
	}
	return ys
}
