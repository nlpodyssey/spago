// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linearregression

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

type LinearRegression struct {
	W *nn.Param `type:"weights"`
}

func NewLinearRegression(in, out int) *LinearRegression {
	return &LinearRegression{
		W: nn.NewParam(mat.NewEmptyDense(out, in)),
	}
}

var _ nn.Processor = &Processor{}

type Processor struct {
	nn.BaseProcessor
	w ag.Node
}

func (m *LinearRegression) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		w: g.NewWrap(m.W),
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("linearregression: invalid init options")
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.Graph.Mul(p.w, x)
	}
	return ys
}
