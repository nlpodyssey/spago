// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sqrdist

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"io"
	"log"
)

var _ nn.Model = &Model{}

type Model struct {
	B *nn.Param `type:"weights"`
}

func New(in, rank int) *Model {
	return &Model{
		B: nn.NewParam(mat.NewEmptyDense(rank, in)),
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
	b     ag.Node
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model: m,
		opt:   opt,
		g:     g,
		b:     g.NewWrap(m.B),
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("sqrdist: invalid init options")
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

func (p *Processor) forward(x ag.Node) ag.Node {
	bh := nn.Linear(p.g, p.b, x)
	return nn.Linear(p.g, p.g.T(bh), bh)
}
