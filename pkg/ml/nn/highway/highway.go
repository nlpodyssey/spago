// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package highway

import (
	"github.com/saientist/spago/pkg/mat"
	"github.com/saientist/spago/pkg/ml/ag"
	"github.com/saientist/spago/pkg/ml/nn"
	"io"
	"log"
)

var _ nn.Model = &Model{}

type Model struct {
	WIn        *nn.Param `type:"weights"`
	BIn        *nn.Param `type:"biases"`
	WT         *nn.Param `type:"weights"`
	BT         *nn.Param `type:"biases"`
	Activation ag.OpName
}

func New(in int, activation ag.OpName) *Model {
	return &Model{
		WIn:        nn.NewParam(mat.NewEmptyDense(in, in)),
		BIn:        nn.NewParam(mat.NewEmptyVecDense(in)),
		WT:         nn.NewParam(mat.NewEmptyDense(in, in)),
		BT:         nn.NewParam(mat.NewEmptyVecDense(in)),
		Activation: activation,
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

type Processor struct {
	opt   []interface{}
	model *Model
	g     *ag.Graph
	wIn   ag.Node
	bIn   ag.Node
	wT    ag.Node
	bT    ag.Node
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model: m,
		opt:   opt,
		g:     g,
		wIn:   g.NewWrap(m.WIn),
		bIn:   g.NewWrap(m.BIn),
		wT:    g.NewWrap(m.WT),
		bT:    g.NewWrap(m.BT),
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("highway: invalid init options")
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
	p.init(p.opt)
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.forward(x)
	}
	return ys
}

// t = sigmoid(wT (dot) x + bT)
// h = f(wIn (dot) x + bIn)
// y = t * h + (1 - t) * x
func (p *Processor) forward(x ag.Node) ag.Node {
	t := p.g.Sigmoid(nn.Affine(p.g, p.bT, p.wT, x))
	h := p.g.Invoke(p.model.Activation, nn.Affine(p.g, p.bIn, p.wIn, x))
	y := p.g.Add(p.g.Prod(t, h), p.g.Prod(p.g.ReverseSub(t, p.g.NewScalar(1.0)), x))
	return y
}
