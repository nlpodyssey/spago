// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cnn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/convolution"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	Convolution    *convolution.Model
	FinalLayer     *linear.Model
	maxPoolingRows int
	maxPoolingCols int
}

// NewModel returns a new model with parameters initialized to zeros.
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

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
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
}

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	p.Convolution.SetMode(mode)
	p.FinalLayer.SetMode(mode)
}

// Forward performs the forward step for each input and returns the result.
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
