// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fixnorm

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Reference: "Improving Lexical Choice in Neural Machine Translation" by Toan Q. Nguyen and David Chiang (2018)
// (https://arxiv.org/pdf/1710.01329.pdf)
type Model struct{}

func New() *Model {
	return &Model{}
}

type Processor struct {
	nn.BaseProcessor
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("fixnorm: invalid init options")
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	g := p.Graph
	ys := make([]ag.Node, len(xs))
	eps := g.NewScalar(1e-10)
	for i, x := range xs {
		norm := g.Sqrt(g.ReduceSum(g.Square(x)))
		ys[i] = g.DivScalar(x, g.AddScalar(norm, eps))
	}
	return ys
}
