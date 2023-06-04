// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"math"

	"github.com/nlpodyssey/spago/mat"
)

// SparseMaxLoss function implementation, based on https://github.com/gokceneraslan/SparseMax.torch
type SparseMaxLoss[O DualValue] struct {
	x   O
	tau float64    // computed during the forward pass
	y   mat.Matrix // computed during forward pass
}

// NewSparseMaxLoss returns a new SparseMaxLoss Function.
func NewSparseMaxLoss[O DualValue](x O) *SparseMaxLoss[O] {
	return &SparseMaxLoss[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *SparseMaxLoss[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *SparseMaxLoss[O]) Forward() (mat.Matrix, error) {
	v := r.x.Value().Clone()

	zs, cumSumInput, bounds, tau := sparseMaxCommon(v)

	tauSquared := tau * tau
	cumSumInputData := cumSumInput.Data().F64()

	var regTerm float64
	for i, zsv := range zs.Data().F64() {
		if bounds[i] > cumSumInputData[i] {
			regTerm += zsv*zsv - tauSquared
		}
	}

	regTerm = regTerm*0.5 + 0.5
	v.SubScalarInPlace(regTerm)

	r.y = v
	r.tau = tau
	return v, nil
}

// Backward computes the backward pass.
func (r *SparseMaxLoss[O]) Backward(gy mat.Matrix) error {
	if r.x.RequiresGrad() {
		tau := r.tau
		gySum := gy.Sum().Scalar().F64()

		sparseMax := r.x.Value().Apply(func(_, _ int, v float64) float64 {
			return math.Max(0, v-tau) * gySum
		})

		gx := gy.Sub(sparseMax)
		r.x.AccGrad(gx)
	}
	return nil
}
