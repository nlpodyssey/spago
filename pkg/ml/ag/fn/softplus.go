// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &SoftPlus{}

// SoftPlus function: f(x) = 1 / β ∗ log(1 + exp(β ∗ x))
type SoftPlus struct {
	x         Operand
	beta      Operand
	threshold Operand
}

// NewSoftPlus returns a new SoftPlus Function.
func NewSoftPlus(x, beta, threshold Operand) *SoftPlus {
	return &SoftPlus{x: x, beta: beta, threshold: threshold}
}

// Forward computes the output of the function.
func (r *SoftPlus) Forward() mat.Matrix {
	y := mat.GetDenseWorkspace(r.x.Value().Dims())
	y.ApplyWithAlpha(softPlus, r.x.Value(), r.beta.Value().Scalar(), r.threshold.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *SoftPlus) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDenseWorkspace(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.ApplyWithAlpha(softPlusDeriv, r.x.Value(), r.beta.Value().Scalar(), r.threshold.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
