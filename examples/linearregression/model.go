// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linearregression

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &LinearRegression{}
	_ nn.Processor = &Processor{}
)

type LinearRegression struct {
	W *nn.Param `type:"weights"`
}

func NewLinearRegression(in, out int) *LinearRegression {
	return &LinearRegression{
		W: nn.NewParam(mat.NewEmptyDense(out, in)),
	}
}

type Processor struct {
	nn.BaseProcessor
	w ag.Node
}

func (m *LinearRegression) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		w: g.NewWrap(m.W),
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.Graph.Mul(p.w, x)
	}
	return ys
}
