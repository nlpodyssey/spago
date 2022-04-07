// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// SparseMaxLoss function implementation, based on https://github.com/gokceneraslan/SparseMax.torch
type SparseMaxLoss[T mat.DType, O Operand[T]] struct {
	x        O
	tau      T             // computed during the forward pass
	y        mat.Matrix[T] // computed during forward pass
	operands []O
}

// NewSparseMaxLoss returns a new SparseMaxLoss Function.
func NewSparseMaxLoss[T mat.DType, O Operand[T]](x O) *SparseMaxLoss[T, O] {
	return &SparseMaxLoss[T, O]{
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *SparseMaxLoss[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *SparseMaxLoss[T, O]) Forward() mat.Matrix[T] {
	v := r.x.Value().Clone()

	zs, cumSumInput, bounds, tau := sparseMaxCommon(v)
	defer mat.ReleaseMatrix(zs)
	defer mat.ReleaseMatrix(cumSumInput)

	tauSquared := tau * tau
	cumSumInputData := cumSumInput.Data()

	var regTerm T = 0.0
	for i, zsv := range zs.Data() {
		if bounds[i] > cumSumInputData[i] {
			regTerm += zsv*zsv - tauSquared
		}
	}

	regTerm = regTerm*0.5 + 0.5
	v.SubScalarInPlace(regTerm)

	r.y = v
	r.tau = tau
	return v
}

// Backward computes the backward pass.
func (r *SparseMaxLoss[T, O]) Backward(gy mat.Matrix[T]) {
	if r.x.RequiresGrad() {
		tau := r.tau
		gySum := gy.Sum()

		sparseMax := r.x.Value().Apply(func(_, _ int, v T) T {
			return mat.Max(0, v-tau) * gySum
		})
		defer mat.ReleaseMatrix(sparseMax)

		gx := gy.Sub(sparseMax)
		defer mat.ReleaseMatrix(gx)

		r.x.AccGrad(gx)
	}
}
