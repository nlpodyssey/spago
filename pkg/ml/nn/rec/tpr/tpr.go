// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tpr

import (
	"github.com/saientist/spago/pkg/mat"
	"github.com/saientist/spago/pkg/ml/ag"
	"github.com/saientist/spago/pkg/ml/nn"
	"io"
	"log"
)

var _ nn.Model = &Model{}

type Model struct {
	WInS *nn.Param `type:"weights"`
	WInR *nn.Param `type:"weights"`

	WRecS *nn.Param `type:"weights"`
	WRecR *nn.Param `type:"weights"`

	BS *nn.Param `type:"biases"`
	BR *nn.Param `type:"biases"`

	S *nn.Param `type:"weights"`
	R *nn.Param `type:"weights"`
}

func New(in, nSymbols, dSymbols, nRoles, dRoles int) *Model {
	return &Model{
		WInS:  nn.NewParam(mat.NewEmptyDense(nSymbols, in)),
		WInR:  nn.NewParam(mat.NewEmptyDense(nRoles, in)),
		WRecS: nn.NewParam(mat.NewEmptyDense(nSymbols, dRoles*dSymbols)),
		WRecR: nn.NewParam(mat.NewEmptyDense(nRoles, dRoles*dSymbols)),
		BS:    nn.NewParam(mat.NewEmptyVecDense(nSymbols)),
		BR:    nn.NewParam(mat.NewEmptyVecDense(nRoles)),
		S:     nn.NewParam(mat.NewEmptyDense(dSymbols, nSymbols)),
		R:     nn.NewParam(mat.NewEmptyDense(dRoles, nRoles)),
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
	AR ag.Node
	AS ag.Node
	S  ag.Node
	R  ag.Node
	Y  ag.Node
}

type InitHidden struct {
	*State
}

type Processor struct {
	opt    []interface{}
	model  *Model
	g      *ag.Graph
	wInS   ag.Node
	wInR   ag.Node
	wRecS  ag.Node
	wRecR  ag.Node
	bS     ag.Node
	bR     ag.Node
	s      ag.Node
	r      ag.Node
	States []*State
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:  m,
		States: nil,
		opt:    opt,
		g:      g,
		wInS:   g.NewWrap(m.WInS),
		wInR:   g.NewWrap(m.WInR),
		wRecS:  g.NewWrap(m.WRecS),
		wRecR:  g.NewWrap(m.WRecR),
		bS:     g.NewWrap(m.BS),
		bR:     g.NewWrap(m.BR),
		s:      g.NewWrap(m.S),
		r:      g.NewWrap(m.R),
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

// aR = Sigmoid(wInR (dot) x + bR + wRecR (dot) yPrev)
// aS = Sigmoid(wInS (dot) x + bS + wRecS (dot) yPrev)
// r = embR (dot) aR
// s = embS (dot) aS
// b = s (dot) rT
// y = vec(b)
func (p *Processor) forward(x ag.Node) (st *State) {
	sPrev := p.LastState()
	var yPrev ag.Node
	if sPrev != nil {
		yPrev = sPrev.Y
	}
	st = new(State)
	st.AR = p.g.Sigmoid(nn.Affine(p.g, p.bR, p.wInR, x, p.wRecR, yPrev))
	st.AS = p.g.Sigmoid(nn.Affine(p.g, p.bS, p.wInS, x, p.wRecS, yPrev))
	st.R = nn.Linear(p.g, p.r, st.AR)
	st.S = nn.Linear(p.g, p.s, st.AS)
	b := nn.Linear(p.g, st.S, p.g.T(st.R))
	st.Y = p.g.Vec(b)
	return
}
