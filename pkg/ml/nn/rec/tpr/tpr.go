// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tpr

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	WInS  *nn.Param `type:"weights"`
	WInR  *nn.Param `type:"weights"`
	WRecS *nn.Param `type:"weights"`
	WRecR *nn.Param `type:"weights"`
	BS    *nn.Param `type:"biases"`
	BR    *nn.Param `type:"biases"`
	S     *nn.Param `type:"weights"`
	R     *nn.Param `type:"weights"`
}

// New returns a new model with parameters initialized to zeros.
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

type State struct {
	AR ag.Node
	AS ag.Node
	S  ag.Node
	R  ag.Node
	Y  ag.Node
}

type Processor struct {
	nn.BaseProcessor
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

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	g := ctx.Graph
	p := &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		wInS:   g.NewWrap(m.WInS),
		wInR:   g.NewWrap(m.WInR),
		wRecS:  g.NewWrap(m.WRecS),
		wRecR:  g.NewWrap(m.WRecR),
		bS:     g.NewWrap(m.BS),
		bR:     g.NewWrap(m.BR),
		s:      g.NewWrap(m.S),
		r:      g.NewWrap(m.R),
		States: nil,
	}
	return p
}

func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("tpr: the initial state must be set before any input")
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
	g := p.Graph
	st.AR = g.Sigmoid(nn.Affine(g, p.bR, p.wInR, x, p.wRecR, yPrev))
	st.AS = g.Sigmoid(nn.Affine(g, p.bS, p.wInS, x, p.wRecS, yPrev))
	st.R = g.Mul(p.r, st.AR)
	st.S = g.Mul(p.s, st.AS)
	b := g.Mul(st.S, g.T(st.R))
	st.Y = g.Vec(b)
	return
}
