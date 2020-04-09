// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Implementation of the MIST (MIxed hiSTory) recurrent network as described in "Analyzing and Exploiting NARX Recurrent
Neural Networks for Long-Term Dependencies" by Di Pietro et al., 2018 (https://arxiv.org/pdf/1702.07805.pdf).
*/
package mist

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"io"
	"log"
	"math"
)

var _ nn.Model = &Model{}

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

func (m *Model) ForEachParam(callback func(param *nn.Param)) {
	nn.ForEachParam(m, callback)
}

func (m *Model) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}

func (m *Model) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

type State struct {
	Y ag.Node
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

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:  m,
		mode:   nn.Training,
		States: nil,
		opt:    opt,
		g:      g,
		wx:     g.NewWrap(m.Wx),
		wh:     g.NewWrap(m.Wh),
		b:      g.NewWrap(m.B),
		wax:    g.NewWrap(m.Wax),
		wah:    g.NewWrap(m.Wah),
		ba:     g.NewWrap(m.Ba),
		wrx:    g.NewWrap(m.Wrx),
		wrh:    g.NewWrap(m.Wrh),
		br:     g.NewWrap(m.Br),
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
			log.Fatal("mist: invalid init option")
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

func (p *Processor) forward(x ag.Node) (s *State) {
	s = new(State)
	yPrev := p.yPrev()
	a := p.g.Softmax(nn.Affine(p.g, p.ba, p.wax, x, p.wah, yPrev))
	r := p.g.Sigmoid(nn.Affine(p.g, p.br, p.wrx, x, p.wrh, yPrev)) // TODO: evaluate whether to calculate this only in case of previous states
	s.Y = p.g.Tanh(nn.Affine(p.g, p.b, p.wx, x, p.wh, p.tryProd(r, p.weightHistory(a))))
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
	var sum ag.Node
	n := len(p.States)
	for i := 0; i < p.model.nd; i++ {
		k := int(math.Pow(2.0, float64(i))) // base-2 exponential delay
		if k <= n {
			sum = p.g.Add(sum, p.g.ProdScalar(p.States[n-k].Y, p.g.AtVec(a, i)))
		}
	}
	return sum
}

// tryProd returns the product if 'a' and 'b' are not nil, otherwise nil
func (p *Processor) tryProd(a, b ag.Node) ag.Node {
	if a != nil && b != nil {
		return p.g.Prod(a, b)
	}
	return nil
}
