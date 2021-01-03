// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package batchnorm

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	W        nn.Param `spago:"type:weights"`
	B        nn.Param `spago:"type:biases"`
	Mean     nn.Param `spago:"type:undefined"`
	StdDev   nn.Param `spago:"type:undefined"`
	Momentum nn.Param `spago:"type:undefined"`
}

const defaultMomentum = 0.9

// NewWithMomentum returns a new model with supplied size and momentum.
func NewWithMomentum(size int, momentum mat.Float) *Model {
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

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	if m.Mode() == nn.Training {
		return m.forwardTraining(xs)
	}
	return m.forwardInference(xs)
}

func (m *Model) forwardTraining(xs []ag.Node) []ag.Node {
	g := m.Graph()
	meanVector := m.mean(xs)
	devVector := m.stdDev(meanVector, xs)
	m.updateBatchNormParameters(meanVector.Value(), devVector.Value())
	return m.process(g, xs, devVector, meanVector)
}

func (m *Model) process(g *ag.Graph, xs []ag.Node, devVector ag.Node, meanVector ag.Node) []ag.Node {
	devVector = g.Div(m.W, g.AddScalar(devVector, g.NewScalar(1e-10)))
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = g.Add(g.Prod(g.Sub(x, meanVector), devVector), m.B)
	}
	return ys
}

func (m *Model) updateBatchNormParameters(meanVector, devVector mat.Matrix) {
	momentum := m.Momentum.Value().Scalar()

	m.Mean.ReplaceValue(
		m.Mean.Value().ProdScalar(momentum).Add(meanVector.ProdScalar(1.0 - momentum)))

	m.StdDev.ReplaceValue(
		m.StdDev.Value().ProdScalar(momentum).Add(devVector.ProdScalar(1.0 - momentum)))
}

func (m *Model) forwardInference(xs []ag.Node) []ag.Node {
	g := m.Graph()
	meanVector := g.NewWrapNoGrad(m.Mean)
	devVector := g.NewWrapNoGrad(m.StdDev)
	return m.process(g, xs, devVector, meanVector)
}

// Mean computes the mean of the input.
func (m *Model) mean(xs []ag.Node) ag.Node {
	g := m.Graph()
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = g.Add(sumVector, xs[i])
	}
	return g.DivScalar(sumVector, g.NewScalar(mat.Float(len(xs))+1e-10))
}

// StdDev computes the standard deviation of the input.
func (m *Model) stdDev(meanVector ag.Node, xs []ag.Node) ag.Node {
	g := m.Graph()
	devVector := g.NewVariable(meanVector.Value().ZerosLike(), false)
	for _, x := range xs {
		diffVector := g.Square(g.Sub(meanVector, x))
		devVector = g.Add(devVector, diffVector)
	}
	devVector = g.Sqrt(g.DivScalar(devVector, g.NewScalar(mat.Float(len(xs))+1e-10)))
	return devVector
}
