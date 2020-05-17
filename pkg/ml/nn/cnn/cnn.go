// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cnn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/convolution"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	Convolution    *convolution.Model
	FinalLayer     *linear.Model
	maxPoolingRows int
	maxPoolingCols int
}

func NewModel(convolution *convolution.Model, maxPoolingRows, maxPoolingCols int, finalLayer *linear.Model) *Model {
	return &Model{
		Convolution:    convolution,
		FinalLayer:     finalLayer,
		maxPoolingRows: maxPoolingRows,
		maxPoolingCols: maxPoolingCols,
	}
}

type Processor struct {
	nn.BaseProcessor
	maxPoolingRows int
	maxPoolingCols int
	Convolution    nn.Processor
	FinalLayer     nn.Processor
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		maxPoolingRows: m.maxPoolingRows,
		maxPoolingCols: m.maxPoolingCols,
		Convolution:    m.Convolution.NewProc(g),
		FinalLayer:     m.FinalLayer.NewProc(g),
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("convolution: invalid init options")
	}
}

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	p.Convolution.SetMode(mode)
	p.FinalLayer.SetMode(mode)
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	filters := p.Convolution.Forward(xs...)
	poolingFilters := p.maxPooling(filters...)
	concatFilters := p.Graph.Concat(p.vectorize(poolingFilters...)...)
	return p.FinalLayer.Forward(concatFilters)
}

func (p *Processor) maxPooling(xs ...ag.Node) []ag.Node {
	ret := make([]ag.Node, len(xs))
	for i, x := range xs {
		ret[i] = p.Graph.MaxPooling(x, p.maxPoolingRows, p.maxPoolingCols)
	}
	return ret
}

func (p *Processor) vectorize(xs ...ag.Node) []ag.Node {
	ret := make([]ag.Node, len(xs))
	for i, x := range xs {
		ret[i] = p.Graph.Vec(x)
	}
	return ret
}
