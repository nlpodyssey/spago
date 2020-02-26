// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package batchnorm

import (
	"io"
	"log"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/ag"
	"saientist.dev/spago/pkg/ml/nn"
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

func (m *Model) ForEachParam(callback func(param *nn.Param)) {
	nn.ForEachParam(m, callback)
}

func (m *Model) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

func (m *Model) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}

type Processor struct {
	opt   []interface{}
	model *Model
	g     *ag.Graph
	w     ag.Node
	b     ag.Node
}

func (m *Model) NewProc(g *ag.Graph, opt ...interface{}) nn.Processor {
	p := &Processor{
		model: m,
		opt:   opt,
		g:     g,
		w:     g.NewWrap(m.W),
		b:     g.NewWrap(m.B),
	}
	p.init(opt)
	return p
}

func (p *Processor) init(opt []interface{}) {
	if len(opt) > 0 {
		log.Fatal("norm: invalid init options")
	}
}

func (p *Processor) Model() nn.Model {
	return p.model
}

func (p *Processor) Graph() *ag.Graph {
	return p.g
}

func (p *Processor) RequiresFullSeq() bool {
	return true
}

func (p *Processor) Reset() {
	p.init(p.opt)
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	meanVector := p.Mean(xs)
	devVector := p.StdDev(meanVector, xs)
	devVector = p.g.Div(p.w, devVector)
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.g.Add(p.g.Prod(p.g.Sub(x, meanVector), devVector), p.b)
	}
	return ys
}

func (p *Processor) Mean(xs []ag.Node) ag.Node {
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = p.g.Add(sumVector, xs[i])
	}
	return p.g.DivScalar(sumVector, p.g.NewScalar(float64(len(xs))+1e-10))
}

func (p *Processor) StdDev(meanVector ag.Node, xs []ag.Node) ag.Node {
	devVector := ag.Node(p.g.NewVariable(meanVector.Value().ZerosLike(), false))
	for _, x := range xs {
		diffVector := p.g.Square(p.g.Sub(meanVector, x))
		devVector = p.g.Add(devVector, diffVector)
	}
	devVector = p.g.Sqrt(p.g.DivScalar(devVector, p.g.NewScalar(float64(len(xs))+1e-10)))
	return devVector
}
