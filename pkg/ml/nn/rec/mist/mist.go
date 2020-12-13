// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package mist provides an implementation of the MIST (MIxed hiSTory) recurrent network as
described in "Analyzing and Exploiting NARX Recurrent Neural Networks for Long-Term Dependencies"
by Di Pietro et al., 2018 (https://arxiv.org/pdf/1702.07805.pdf).
*/
package mist

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
	"math"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	Wx  *nn.Param `type:"weights"`
	Wh  *nn.Param `type:"weights"`
	B   *nn.Param `type:"biases"`
	Wax *nn.Param `type:"weights"`
	Wah *nn.Param `type:"weights"`
	Ba  *nn.Param `type:"biases"`
	Wrx *nn.Param `type:"weights"`
	Wrh *nn.Param `type:"weights"`
	Br  *nn.Param `type:"biases"`
	nd  int       // number of delays
}

// New returns a new model with parameters initialized to zeros.
func New(in, out, numberOfDelays int) *Model {
	return &Model{
		Wx:  nn.NewParam(mat.NewEmptyDense(out, in)),
		Wh:  nn.NewParam(mat.NewEmptyDense(out, out)),
		B:   nn.NewParam(mat.NewEmptyVecDense(out)),
		Wax: nn.NewParam(mat.NewEmptyDense(out, in)),
		Wah: nn.NewParam(mat.NewEmptyDense(out, out)),
		Ba:  nn.NewParam(mat.NewEmptyVecDense(out)),
		Wrx: nn.NewParam(mat.NewEmptyDense(out, in)),
		Wrh: nn.NewParam(mat.NewEmptyDense(out, out)),
		Br:  nn.NewParam(mat.NewEmptyVecDense(out)),
		nd:  numberOfDelays,
	}
}

type State struct {
	Y ag.Node
}

type Processor struct {
	nn.BaseProcessor
	nd     int
	wx     ag.Node
	wh     ag.Node
	b      ag.Node
	wax    ag.Node
	wah    ag.Node
	ba     ag.Node
	wrx    ag.Node
	wrh    ag.Node
	br     ag.Node
	States []*State
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
		nd:     m.nd,
		wx:     g.NewWrap(m.Wx),
		wh:     g.NewWrap(m.Wh),
		b:      g.NewWrap(m.B),
		wax:    g.NewWrap(m.Wax),
		wah:    g.NewWrap(m.Wah),
		ba:     g.NewWrap(m.Ba),
		wrx:    g.NewWrap(m.Wrx),
		wrh:    g.NewWrap(m.Wrh),
		br:     g.NewWrap(m.Br),
		States: nil,
	}
}

func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("mist: the initial state must be set before any input")
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

func (p *Processor) forward(x ag.Node) (s *State) {
	g := p.Graph
	s = new(State)
	yPrev := p.yPrev()
	a := g.Softmax(nn.Affine(g, p.ba, p.wax, x, p.wah, yPrev))
	r := g.Sigmoid(nn.Affine(g, p.br, p.wrx, x, p.wrh, yPrev)) // TODO: evaluate whether to calculate this only in case of previous states
	s.Y = g.Tanh(nn.Affine(g, p.b, p.wx, x, p.wh, p.tryProd(r, p.weightHistory(a))))
	return
}

func (p *Processor) yPrev() ag.Node {
	var yPrev ag.Node
	s := p.LastState()
	if s != nil {
		yPrev = s.Y
	}
	return yPrev
}

func (p *Processor) weightHistory(a ag.Node) ag.Node {
	g := p.Graph
	var sum ag.Node
	n := len(p.States)
	for i := 0; i < p.nd; i++ {
		k := int(math.Pow(2.0, float64(i))) // base-2 exponential delay
		if k <= n {
			sum = g.Add(sum, g.ProdScalar(p.States[n-k].Y, g.AtVec(a, i)))
		}
	}
	return sum
}

// tryProd returns the product if 'a' and 'b' are not nil, otherwise nil
func (p *Processor) tryProd(a, b ag.Node) ag.Node {
	if a != nil && b != nil {
		return p.Graph.Prod(a, b)
	}
	return nil
}
