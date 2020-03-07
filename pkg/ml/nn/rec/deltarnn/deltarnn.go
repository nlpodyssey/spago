// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deltarnn

import (
	"io"
	"log"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/ag"
	"saientist.dev/spago/pkg/ml/nn"
)

var _ nn.Model = &Model{}

type Model struct {
	W     *nn.Param `type:"weights"`
	WRec  *nn.Param `type:"weights"`
	B     *nn.Param `type:"biases"`
	BPart *nn.Param `type:"biases"`
	Alpha *nn.Param `type:"weights"`
	Beta1 *nn.Param `type:"weights"`
	Beta2 *nn.Param `type:"weights"`
}

func New(in, out int) *Model {
	return &Model{
		W:     nn.NewParam(mat.NewEmptyDense(out, in)),
		WRec:  nn.NewParam(mat.NewEmptyDense(out, out)),
		B:     nn.NewParam(mat.NewEmptyVecDense(out)),
		BPart: nn.NewParam(mat.NewEmptyVecDense(out)),
		Alpha: nn.NewParam(mat.NewEmptyVecDense(out)),
		Beta1: nn.NewParam(mat.NewEmptyVecDense(out)),
		Beta2: nn.NewParam(mat.NewEmptyVecDense(out)),
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
	D1 ag.Node
	D2 ag.Node
	C  ag.Node
	P  ag.Node
	Y  ag.Node
}

type InitHidden struct {
	*State
}

type Processor struct {
	opt    []interface{}
	model  *Model
	g      *ag.Graph
	w      ag.Node
	wRec   ag.Node
	b      ag.Node
	bPart  ag.Node
	alpha  ag.Node
	beta1  ag.Node
	beta2  ag.Node
	States []*State
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:  m,
		States: nil,
		opt:    opt,
		g:      g,
		w:      g.NewWrap(m.W),
		wRec:   g.NewWrap(m.WRec),
		b:      g.NewWrap(m.B),
		bPart:  g.NewWrap(m.BPart),
		alpha:  g.NewWrap(m.Alpha),
		beta1:  g.NewWrap(m.Beta1),
		beta2:  g.NewWrap(m.Beta2),
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

// d1 = beta1 * w (dot) x + beta2 * wRec (dot) yPrev
// d2 = alpha * w (dot) x * wRec (dot) yPrev
// c = tanh(d1 + d2 + bc)
// p = sigmoid(w (dot) x + bp)
// y = f(p * c + (1 - p) * yPrev)
func (p *Processor) forward(x ag.Node) (s *State) {
	s = new(State)
	yPrev := p.prev()
	wx := nn.Linear(p.g, p.w, x)
	if yPrev == nil {
		s.D1 = p.g.Prod(p.beta1, wx)
		s.C = p.g.Tanh(p.g.Add(s.D1, p.b))
		s.P = p.g.Sigmoid(p.g.Add(wx, p.bPart))
		s.Y = p.g.Tanh(p.g.Prod(s.P, s.C))
	} else {
		wyRec := nn.Linear(p.g, p.wRec, yPrev)
		s.D1 = p.g.Add(p.g.Prod(p.beta1, wx), p.g.Prod(p.beta2, wyRec))
		s.D2 = p.g.Prod(p.g.Prod(p.alpha, wx), wyRec)
		s.C = p.g.Tanh(p.g.Add(p.g.Add(s.D1, s.D2), p.b))
		s.P = p.g.Sigmoid(p.g.Add(wx, p.bPart))
		s.Y = p.g.Tanh(p.g.Add(p.g.Prod(s.P, s.C), p.g.Prod(p.g.ReverseSub(s.P, p.g.NewScalar(1.0)), yPrev)))
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
