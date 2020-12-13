// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package batchnorm

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
	W        *nn.Param `type:"weights"`
	B        *nn.Param `type:"biases"`
	Mean     *nn.Param `type:"undefined"`
	StdDev   *nn.Param `type:"undefined"`
	Momentum *nn.Param `type:"undefined"`
}

const defaultMomentum = 0.9

// NewWithMomentum returns a new model with supplied size and momentum.
func NewWithMomentum(size int, momentum float64) *Model {
	return &Model{
		W:        nn.NewParam(mat.NewInitVecDense(size, 1.0)),
		B:        nn.NewParam(mat.NewEmptyVecDense(size)),
		Mean:     nn.NewParam(mat.NewEmptyVecDense(size), nn.RequiresGrad(false)),
		StdDev:   nn.NewParam(mat.NewEmptyVecDense(size), nn.RequiresGrad(false)),
		Momentum: nn.NewParam(mat.NewScalar(momentum), nn.RequiresGrad(false)),
	}
}

// New returns a new model with the supplied size and default momentum
func New(size int) *Model {
	return NewWithMomentum(size, defaultMomentum)
}

type Processor struct {
	nn.BaseProcessor
	w     ag.Node
	b     ag.Node
	model *Model
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		w:     ctx.Graph.NewWrap(m.W),
		b:     ctx.Graph.NewWrap(m.B),
		model: m,
	}
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	if p.Mode == nn.Training {
		return p.forwardTraining(xs)
	}
	return p.forwardInference(xs)
}

func (p *Processor) forwardTraining(xs []ag.Node) []ag.Node {
	g := p.Graph
	meanVector := p.Mean(xs)
	devVector := p.StdDev(meanVector, xs)
	p.updateBatchNormParameters(meanVector.Value(), devVector.Value())
	return p.process(g, xs, devVector, meanVector)
}

func (p *Processor) process(g *ag.Graph, xs []ag.Node, devVector ag.Node, meanVector ag.Node) []ag.Node {
	devVector = g.Div(p.w, devVector)
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = g.Add(g.Prod(g.Sub(x, meanVector), devVector), p.b)
	}
	return ys
}

func (p *Processor) updateBatchNormParameters(meanVector, devVector mat.Matrix) {
	momentum := p.model.Momentum.Value().Scalar()

	p.model.Mean.ReplaceValue(
		p.model.Mean.Value().ProdScalar(momentum).Add(meanVector.ProdScalar(1.0 - momentum)))

	p.model.StdDev.ReplaceValue(
		p.model.StdDev.Value().ProdScalar(momentum).Add(devVector.ProdScalar(1.0 - momentum)))
}

func (p *Processor) forwardInference(xs []ag.Node) []ag.Node {
	g := p.Graph
	meanVector := g.NewWrapNoGrad(p.model.Mean)
	devVector := g.NewWrapNoGrad(p.model.StdDev)
	return p.process(g, xs, devVector, meanVector)
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
