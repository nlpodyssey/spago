// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flatten

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model is a parameter-free model used to instantiate a new Processor.
type Model struct{}

// New returns a new model.
// TODO: think about possible configurations
func New() *Model {
	return &Model{}
}

type Processor struct {
	nn.BaseProcessor
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
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	return []ag.Node{p.Graph.Concat(p.vectorize(xs...)...)}
}

func (p *Processor) vectorize(xs ...ag.Node) []ag.Node {
	ret := make([]ag.Node, len(xs))
	for i, x := range xs {
		ret[i] = p.Graph.Vec(x)
	}
	return ret
}
