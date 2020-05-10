// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package adanorm

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Reference: "Understanding and Improving Layer Normalization" by Jingjing Xu, Xu Sun, Zhiyuan Zhang, Guangxiang Zhao, Junyang Lin (2019).
// (https://papers.nips.cc/paper/8689-understanding-and-improving-layer-normalization.pdf)
type Model struct {
	scale float64
}

func New(scale float64) *Model {
	return &Model{scale: scale}
}

type Processor struct {
	opt   []interface{}
	model *Model
	mode  nn.ProcessingMode
	g     *ag.Graph
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model: m,
		mode:  nn.Training,
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

func (p *Processor) Model() nn.Model                { return p.model }
func (p *Processor) Graph() *ag.Graph               { return p.g }
func (p *Processor) RequiresFullSeq() bool          { return true }
func (p *Processor) Mode() nn.ProcessingMode        { return p.mode }
func (p *Processor) SetMode(mode nn.ProcessingMode) { p.mode = mode }

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	meanVectors := p.Mean(xs)
	devVectors := p.StdDev(meanVectors, xs)
	zs := make([]ag.Node, len(xs))
	eps := p.g.NewScalar(1e-10)
	k := p.g.NewScalar(0.1)
	c := p.g.NewScalar(p.model.scale)
	for i, x := range xs {
		y := p.g.DivScalar(p.g.SubScalar(x, meanVectors[i]), p.g.Add(devVectors[i], eps))
		fi := p.g.ProdScalar(p.g.ReverseSub(p.g.ProdScalar(y, k), p.g.NewScalar(1.0)), c)
		zs[i] = p.g.Prod(y, p.g.NewWrapNoGrad(fi)) // detach the gradient of fi and only treat it as a changeable constant in implementation
	}
	return zs
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
