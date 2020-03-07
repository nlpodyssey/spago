// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cnn

import (
	"io"
	"log"
	"saientist.dev/spago/pkg/ml/ag"
	"saientist.dev/spago/pkg/ml/nn"
	"saientist.dev/spago/pkg/ml/nn/convolution"
	"saientist.dev/spago/pkg/ml/nn/perceptron"
)

var _ nn.Model = &Model{}

type Model struct {
	Convolution    *convolution.Model
	FinalLayer     *perceptron.Model
	maxPoolingRows int
	maxPoolingCols int
}

func NewModel(convolution *convolution.Model, maxPoolingRows, maxPoolingCols int, finalLayer *perceptron.Model) *Model {
	return &Model{
		Convolution:    convolution,
		FinalLayer:     finalLayer,
		maxPoolingRows: maxPoolingRows,
		maxPoolingCols: maxPoolingCols,
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
	opt         []interface{}
	model       *Model
	g           *ag.Graph
	Convolution nn.Processor
	Perceptron  nn.Processor
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model:       m,
		Convolution: m.Convolution.NewProc(g),
		Perceptron:  m.FinalLayer.NewProc(g),
		g:           g,
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("convolution: invalid init options")
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
	p.Convolution.Reset()
	p.Perceptron.Reset()
	p.init(p.opt)
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	filters := p.Convolution.Forward(xs...)
	poolingFilters := p.maxPooling(filters...)
	concatFilters := p.g.Concat(p.vectorize(poolingFilters...)...)
	return p.Perceptron.Forward(concatFilters)
}

func (p *Processor) maxPooling(xs ...ag.Node) []ag.Node {
	ret := make([]ag.Node, len(xs))
	for i, x := range xs {
		ret[i] = p.g.MaxPooling(x, p.model.maxPoolingRows, p.model.maxPoolingCols)
	}
	return ret
}

func (p *Processor) vectorize(xs ...ag.Node) []ag.Node {
	ret := make([]ag.Node, len(xs))
	for i, x := range xs {
		ret[i] = p.g.Vec(x)
	}
	return ret
}
