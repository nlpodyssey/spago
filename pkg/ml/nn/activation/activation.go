// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the activation operator and serializable parameters.
type Model struct {
	Activation ag.OpName
	Params     []*nn.Param
}

// TODO: restrict operators to activation functions only; or create a dedicate builder for each activation.
// New returns a new model with parameters initialized to zeros.
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

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	var params []ag.Node
	for _, param := range m.Params {
		params = append(params, g.NewWrap(param))
	}
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		params: params,
	}
}

// Forward performs the the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	activation := p.Model.(*Model).Activation
	for i, x := range xs {
		ys[i] = p.Graph.Invoke(activation, append([]ag.Node{x}, p.params...)...)
	}
	return ys
}
