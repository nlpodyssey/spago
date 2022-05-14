// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package batchnorm

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.Module
	W        nn.Param[T] `spago:"type:weights"`
	B        nn.Param[T] `spago:"type:biases"`
	Mean     nn.Param[T] `spago:"type:undefined"`
	StdDev   nn.Param[T] `spago:"type:undefined"`
	Momentum nn.Param[T] `spago:"type:undefined"`
}

const epsilon = 1e-5
const defaultMomentum = 0.9

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// NewWithMomentum returns a new model with supplied size and momentum.
func NewWithMomentum[T mat.DType](size int, momentum T) *Model[T] {
	return &Model[T]{
		W:        nn.NewParam[T](mat.NewInitVecDense[T](size, epsilon)),
		B:        nn.NewParam[T](mat.NewEmptyVecDense[T](size)),
		Mean:     nn.NewParam[T](mat.NewEmptyVecDense[T](size), nn.RequiresGrad[T](false)),
		StdDev:   nn.NewParam[T](mat.NewEmptyVecDense[T](size), nn.RequiresGrad[T](false)),
		Momentum: nn.NewParam[T](mat.NewScalar[T](momentum), nn.RequiresGrad[T](false)),
	}
}

// New returns a new model with the supplied size and default momentum
func New[T mat.DType](size int) *Model[T] {
	return NewWithMomentum[T](size, defaultMomentum)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	meanVector := ag.StopGrad[T](m.Mean)
	devVector := ag.StopGrad[T](m.StdDev)
	return m.process(xs, devVector, meanVector)
}

// ForwardT performs the forward step for each input node and returns the result.
func (m *Model[T]) ForwardT(xs ...ag.Node[T]) []ag.Node[T] {
	meanVector := m.mean(xs)
	devVector := m.stdDev(meanVector, xs)
	m.updateBatchNormParameters(meanVector.Value(), devVector.Value())
	return m.process(xs, devVector, meanVector)
}

func (m *Model[T]) process(xs []ag.Node[T], devVector ag.Node[T], meanVector ag.Node[T]) []ag.Node[T] {
	devVector = ag.Div[T](m.W, ag.AddScalar(devVector, ag.Constant[T](epsilon)))
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		ys[i] = ag.Add[T](ag.Prod(ag.Sub(x, meanVector), devVector), m.B)
	}
	return ys
}

func (m *Model[T]) updateBatchNormParameters(meanVector, devVector mat.Matrix) {
	momentum := m.Momentum.Value().Scalar().Float64()

	m.Mean.ReplaceValue(
		m.Mean.Value().ProdScalar(momentum).Add(meanVector.ProdScalar(1.0 - momentum)))

	m.StdDev.ReplaceValue(
		m.StdDev.Value().ProdScalar(momentum).Add(devVector.ProdScalar(1.0 - momentum)))
}

// Mean computes the mean of the input.
func (m *Model[T]) mean(xs []ag.Node[T]) ag.Node[T] {
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = ag.Add(sumVector, xs[i])
	}

	return ag.DivScalar(sumVector, ag.NewScalar[T](T(len(xs))+epsilon))
}

// StdDev computes the standard deviation of the input.
func (m *Model[T]) stdDev(meanVector ag.Node[T], xs []ag.Node[T]) ag.Node[T] {
	devVector := ag.NewVariable[T](meanVector.Value().ZerosLike(), false)
	for _, x := range xs {
		diffVector := ag.Square(ag.Sub(meanVector, x))
		devVector = ag.Add(devVector, diffVector)
	}
	devVector = ag.Sqrt(ag.DivScalar(devVector, ag.NewScalar[T](T(len(xs))+epsilon)))
	return devVector
}
