// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layernormsimple

import (
	"github.com/saientist/spago/pkg/ml/ag"
	"github.com/saientist/spago/pkg/ml/nn"
	"io"
	"log"
)

var _ nn.Model = &Model{}

// Reference: "Understanding and Improving Layer Normalization" by Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, Junyang Lin (2019).
// (https://papers.nips.cc/paper/8689-understanding-and-improving-layer-normalization.pdf)
type Model struct{}

func New() *Model {
	return &Model{}
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
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model: m,
		opt:   opt,
		g:     g,
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("layernormsimple: invalid init options")
	}
}

func (p *Processor) Model() nn.Model {
	return p.model
}

func (p *Processor) Graph() *ag.Graph {
	return p.g
}

func (p *Processor) RequiresFullSeq() bool {
	return true
}

func (p *Processor) Reset() {
	p.init(p.opt)
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	meanVectors := p.Mean(xs)
	devVectors := p.StdDev(meanVectors, xs)
	ys := make([]ag.Node, len(xs))
	eps := p.g.NewScalar(1e-10)
	for i, x := range xs {
		ys[i] = p.g.DivScalar(p.g.SubScalar(x, meanVectors[i]), p.g.Add(devVectors[i], eps))
	}
	return ys
}

func (p *Processor) Mean(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.g.ReduceMean(x)
	}
	return ys
}

func (p *Processor) StdDev(meanVectors []ag.Node, xs []ag.Node) []ag.Node {
	devVectors := make([]ag.Node, len(xs))
	for i, x := range xs {
		diffVector := p.g.Square(p.g.SubScalar(x, meanVectors[i]))
		devVectors[i] = p.g.Sqrt(p.g.ReduceMean(diffVector))
	}
	return devVectors
}
