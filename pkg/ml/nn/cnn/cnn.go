// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cnn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/convolution"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/pooling"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	Convolution *convolution.Model
	MaxPooling  *pooling.MaxPooling
	FinalLayer  *linear.Model
}

// NewModel returns a new model with parameters initialized to zeros.
func NewModel(convolution *convolution.Model, pooling *pooling.MaxPooling, finalLayer *linear.Model) *Model {
	return &Model{
		Convolution: convolution,
		MaxPooling:  pooling,
		FinalLayer:  finalLayer,
	}
}

type Processor struct {
	nn.BaseProcessor
	Convolution *convolution.Processor
	Pooling     *pooling.Processor
	FinalLayer  *linear.Processor
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
		Pooling:     m.MaxPooling.NewProc(g).(*pooling.Processor),
		Convolution: m.Convolution.NewProc(g).(*convolution.Processor),
		FinalLayer:  m.FinalLayer.NewProc(g).(*linear.Processor),
	}
}

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	p.Convolution.SetMode(mode)
	p.FinalLayer.SetMode(mode)
	p.Pooling.SetMode(mode)
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	filters := p.Convolution.Forward(xs...)
	poolingFilters := p.Pooling.Forward(filters...)
	concatFilters := p.Graph.Concat(p.vectorize(poolingFilters...)...)
	return p.FinalLayer.Forward(concatFilters)
}

func (p *Processor) vectorize(xs ...ag.Node) []ag.Node {
	ret := make([]ag.Node, len(xs))
	for i, x := range xs {
		ret[i] = p.Graph.Vec(x)
	}
	return ret
}
