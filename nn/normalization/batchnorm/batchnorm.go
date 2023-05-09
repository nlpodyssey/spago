// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package batchnorm

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	W        *nn.Param
	B        *nn.Param
	Mean     *nn.Buffer
	StdDev   *nn.Buffer
	Momentum *nn.Buffer
}

const epsilon = 1e-5
const defaultMomentum = 0.9

func init() {
	gob.Register(&Model{})
}

// NewWithMomentum returns a new model with supplied size and momentum.
func NewWithMomentum[T float.DType](size int, momentum T) *Model {
	return &Model{
		W:        nn.NewParam(mat.NewInitVecDense[T](size, epsilon)),
		B:        nn.NewParam(mat.NewEmptyVecDense[T](size)),
		Mean:     nn.Buf(mat.NewEmptyVecDense[T](size)),
		StdDev:   nn.Buf(mat.NewEmptyVecDense[T](size)),
		Momentum: nn.Const(momentum),
	}
}

// New returns a new model with the supplied size and default momentum
func New[T float.DType](size int) *Model {
	return NewWithMomentum[T](size, defaultMomentum)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.DualValue) []ag.DualValue {
	meanVector := ag.StopGrad(m.Mean)
	devVector := ag.StopGrad(m.StdDev)
	return m.process(xs, devVector, meanVector)
}

// ForwardT performs the forward step for each input node and returns the result.
func (m *Model) ForwardT(xs ...ag.DualValue) []ag.DualValue {
	meanVector := m.mean(xs)
	devVector := m.stdDev(meanVector, xs)
	m.updateBatchNormParameters(meanVector.Value(), devVector.Value())
	return m.process(xs, devVector, meanVector)
}

func (m *Model) process(xs []ag.DualValue, devVector ag.DualValue, meanVector ag.DualValue) []ag.DualValue {
	devVector = ag.Div(m.W, ag.AddScalar(devVector, m.W.Value().NewScalar(epsilon)))
	ys := make([]ag.DualValue, len(xs))
	for i, x := range xs {
		ys[i] = ag.Add(ag.Prod(ag.Sub(x, meanVector), devVector), m.B)
	}
	return ys
}

func (m *Model) updateBatchNormParameters(meanVector, devVector mat.Matrix) {
	momentum := m.Momentum.Scalar().F64()
	m.Mean.ProdScalarInPlace(momentum).AddInPlace(meanVector.ProdScalar(1.0 - momentum))
	m.StdDev.ProdScalarInPlace(momentum).AddInPlace(devVector.ProdScalar(1.0 - momentum))
}

// Mean computes the mean of the input.
func (m *Model) mean(xs []ag.DualValue) ag.DualValue {
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = ag.Add(sumVector, xs[i])
	}

	return ag.DivScalar(sumVector, xs[0].Value().NewScalar(float64(len(xs))+epsilon))
}

// StdDev computes the standard deviation of the input.
func (m *Model) stdDev(meanVector ag.DualValue, xs []ag.DualValue) ag.DualValue {
	devVector := ag.DualValue(meanVector.Value().ZerosLike())
	for _, x := range xs {
		diffVector := ag.Square(ag.Sub(meanVector, x))
		devVector = ag.Add(devVector, diffVector)
	}
	devVector = ag.Sqrt(ag.DivScalar(devVector, xs[0].Value().NewScalar(float64(len(xs))+epsilon)))
	return devVector
}
