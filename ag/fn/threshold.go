// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Threshold function: f(x) = x if x > threshold; k otherwise.
type Threshold[O Operand] struct {
	x         O
	threshold O // scalar
	k         O // scalar
	operands  []O
}

// NewThreshold returns a new Threshold Function.
func NewThreshold[O Operand](x O, threshold, k O) *Threshold[O] {
	return &Threshold[O]{
		x:         x,
		threshold: threshold,
		k:         k,
		operands:  []O{x, threshold, k},
	}
}

// Operands returns the list of operands.
func (r *Threshold[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *Threshold[O]) Forward() mat.Matrix {
	y := r.x.Value().ApplyWithAlpha(
		threshold,
		r.threshold.Value().Scalar().Float64(),
		r.k.Value().Scalar().Float64(),
	)
	return y
}

// Backward computes the backward pass.
func (r *Threshold[O]) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(
			thresholdDeriv,
			r.threshold.Value().Scalar().Float64(),
			r.k.Value().Scalar().Float64(),
		)
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.AccGrad(gx)
	}
}
