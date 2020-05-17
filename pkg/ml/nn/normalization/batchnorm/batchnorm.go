// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package batchnorm

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	W *nn.Param `type:"weights"`
	B *nn.Param `type:"biases"`
}

func New(size int) *Model {
	return &Model{
		W: nn.NewParam(mat.NewEmptyVecDense(size)),
		B: nn.NewParam(mat.NewEmptyVecDense(size)),
	}
}

type Processor struct {
	nn.BaseProcessor
	w ag.Node
	b ag.Node
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		w: g.NewWrap(m.W),
		b: g.NewWrap(m.B),
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("batchnorm: invalid init options")
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	g := p.Graph
	meanVector := p.Mean(xs)
	devVector := p.StdDev(meanVector, xs)
	devVector = g.Div(p.w, devVector)
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = g.Add(g.Prod(g.Sub(x, meanVector), devVector), p.b)
	}
	return ys
}

func (p *Processor) Mean(xs []ag.Node) ag.Node {
	g := p.Graph
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = g.Add(sumVector, xs[i])
	}
	return g.DivScalar(sumVector, g.NewScalar(float64(len(xs))+1e-10))
}

func (p *Processor) StdDev(meanVector ag.Node, xs []ag.Node) ag.Node {
	g := p.Graph
	devVector := g.NewVariable(meanVector.Value().ZerosLike(), false)
	for _, x := range xs {
		diffVector := g.Square(g.Sub(meanVector, x))
		devVector = g.Add(devVector, diffVector)
	}
	devVector = g.Sqrt(g.DivScalar(devVector, g.NewScalar(float64(len(xs))+1e-10)))
	return devVector
}
