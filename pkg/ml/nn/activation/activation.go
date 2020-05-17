// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	Activation ag.OpName
	Params     []*nn.Param
}

// TODO: restrict operators to activation functions only; or create a dedicate builder for each activation.
func New(activation ag.OpName, params ...*nn.Param) *Model {
	return &Model{
		Activation: activation,
		Params:     params,
	}
}

type Processor struct {
	nn.BaseProcessor
	params []ag.Node
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	var params []ag.Node
	for _, param := range m.Params {
		params = append(params, g.NewWrap(param))
	}
	p := &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		params: params,
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("activation: invalid init options")
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	activation := p.Model.(*Model).Activation
	for i, x := range xs {
		ys[i] = p.Graph.Invoke(activation, append([]ag.Node{x}, p.params...)...)
	}
	return ys
}
