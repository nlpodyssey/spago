// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package horn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils"
	"io"
	"log"
	"math"
)

var _ nn.Model = &Model{}

// Higher Order Recurrent Neural Networks
type Model struct {
	W    *nn.Param   `type:"weights"`
	WRec []*nn.Param `type:"weights"`
	B    *nn.Param   `type:"biases"`
}

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
	w      ag.Node
	wRec   []ag.Node
	b      ag.Node
	States []*State
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	wRec := make([]ag.Node, len(m.WRec), len(m.WRec))
	for i, param := range m.WRec {
		wRec[i] = g.NewWrap(param)
	}
	p := &Processor{
		model:  m,
		mode:   nn.Training,
		States: nil,
		opt:    opt,
		g:      g,
		w:      g.NewWrap(m.W),
		wRec:   wRec,
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
			log.Fatal("horn: invalid init option")
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

func (p *Processor) forward(x ag.Node) (s *State) {
	s = new(State)
	h := nn.Affine(p.g, append([]ag.Node{p.b, p.w, x}, p.feedback()...)...)
	s.Y = p.g.Tanh(h)
	return
}

func (p *Processor) feedback() []ag.Node {
	var ys []ag.Node
	n := len(p.States)
	for i := 0; i < utils.MinInt(len(p.wRec), n); i++ {
		alpha := p.g.NewScalar(math.Pow(0.6, float64(i+1)))
		ys = append(ys, p.wRec[i], p.g.ProdScalar(p.States[n-1-i].Y, alpha))
	}
	return ys
}
