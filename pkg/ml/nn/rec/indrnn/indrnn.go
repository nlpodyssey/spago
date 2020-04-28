// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package indrnn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"io"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	W          *nn.Param `type:"weights"`
	WRec       *nn.Param `type:"weights"`
	B          *nn.Param `type:"biases"`
	Activation ag.OpName // output activation
}

func New(in, out int, activation ag.OpName) *Model {
	return &Model{
		W:          nn.NewParam(mat.NewEmptyDense(out, in)),
		WRec:       nn.NewParam(mat.NewEmptyVecDense(out)),
		B:          nn.NewParam(mat.NewEmptyVecDense(out)),
		Activation: activation,
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

type Processor struct {
	opt    []interface{}
	model  *Model
	mode   nn.ProcessingMode
	g      *ag.Graph
	w      ag.Node
	wRec   ag.Node
	b      ag.Node
	States []*State
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:  m,
		mode:   nn.Training,
		States: nil,
		opt:    opt,
		g:      g,
		w:      g.NewWrap(m.W),
		wRec:   g.NewWrap(m.WRec),
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

// y = f(w (dot) x + wRec * yPrev + b)
func (p *Processor) forward(x ag.Node) (s *State) {
	s = new(State)
	yPrev := p.prev()
	h := nn.Affine(p.g, p.b, p.w, x)
	if yPrev != nil {
		h = p.g.Add(h, p.g.Prod(p.wRec, yPrev))
	}
	s.Y = p.g.Invoke(p.model.Activation, h)
	return
}

func (p *Processor) prev() (yPrev ag.Node) {
	s := p.LastState()
	if s != nil {
		yPrev = s.Y
	}
	return
}
