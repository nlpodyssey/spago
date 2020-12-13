// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stack

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	Layers []nn.Model
}

// New returns a new model.
func New(layers ...nn.Model) *Model {
	return &Model{
		Layers: layers,
	}
}

// Make makes a new model by obtaining each layer with a callback.
func Make(size int, callback func(i int) nn.Model) *Model {
	layers := make([]nn.Model, size)
	for i := 0; i < size; i++ {
		layers[i] = callback(i)
	}
	return &Model{
		Layers: layers,
	}
}

func (m *Model) LastLayer() nn.Model {
	return m.Layers[len(m.Layers)-1]
}

type Processor struct {
	nn.BaseProcessor
	Layers []nn.Processor
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	procLayers := make([]nn.Processor, len(m.Layers))
	for i, layer := range m.Layers {
		procLayers[i] = layer.NewProc(ctx)
	}
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: requiresFullSeq(procLayers),
		},
		Layers: procLayers,
	}
}

func requiresFullSeq(ps []nn.Processor) bool {
	for _, p := range ps {
		if p.RequiresFullSeq() {
			return true
		}
	}
	return false
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	if p.RequiresFullSeq() {
		return p.fullSeqForward(xs)
	}
	return p.incrementalForward(xs)
}

func (p *Processor) fullSeqForward(xs []ag.Node) []ag.Node {
	ys := p.Layers[0].Forward(xs...)
	for i := 1; i < len(p.Layers); i++ {
		ys = p.Layers[i].Forward(ys...)
	}
	return ys
}

func (p *Processor) incrementalForward(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.singleForward(x)
	}
	return ys
}

func (p *Processor) singleForward(x ag.Node) ag.Node {
	y := x
	for _, layer := range p.Layers {
		y = layer.Forward(y)[0]
	}
	return y
}
