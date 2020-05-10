// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stack

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"io"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	Layers []nn.Model
}

func New(layers ...nn.Model) *Model {
	return &Model{
		Layers: layers,
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

func (m *Model) LastLayer() nn.Model {
	return m.Layers[len(m.Layers)-1]
}

type Processor struct {
	opt             []interface{}
	model           *Model
	mode            nn.ProcessingMode
	g               *ag.Graph
	Layers          []nn.Processor
	requiresFullSeq bool
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	if opt != nil && len(opt) > 0 {
		if len(opt) != len(m.Layers) {
			log.Fatal("stack: the options must be grouped in lists of options parallel to the layers")
		}
	}
	ps := make([]nn.Processor, len(m.Layers))
	for i, layer := range m.Layers {
		var layerOpt []interface{}
		if opt != nil {
			layerOpt = opt[i].([]interface{})
		}
		ps[i] = layer.NewProc(g, layerOpt...)
	}
	p := &Processor{
		model:           m,
		mode:            nn.Training,
		Layers:          ps,
		opt:             opt,
		g:               g,
		requiresFullSeq: requiresFullSeq(ps),
	}
	return p
}

func (p *Processor) Model() nn.Model         { return p.model }
func (p *Processor) Graph() *ag.Graph        { return p.g }
func (p *Processor) RequiresFullSeq() bool   { return p.requiresFullSeq }
func (p *Processor) Mode() nn.ProcessingMode { return p.mode }

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.mode = mode
	for _, layer := range p.Layers {
		layer.SetMode(mode)
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

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	if p.RequiresFullSeq() {
		return p.fullSeqForward(xs)
	} else {
		return p.incrementalForward(xs)
	}
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
