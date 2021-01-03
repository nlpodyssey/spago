// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &CELU{}

// CELU is an operator to perform the CELU activation.
// CELU(x) = max(0,x) + min(0,α ∗ (exp(x/α) − 1))
type CELU struct {
	x     Operand
	alpha Operand // scalar
}

// NewCELU returns a new CELU Function.
func NewCELU(x, alpha Operand) *CELU {
	return &CELU{x: x, alpha: alpha}
}

// Forward computes the output of the function.
func (r *CELU) Forward() mat.Matrix {
	y := mat.GetDenseWorkspace(r.x.Value().Dims())
	y.ApplyWithAlpha(celu, r.x.Value(), r.alpha.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *CELU) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDenseWorkspace(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.ApplyWithAlpha(celuDeriv, r.x.Value(), r.alpha.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
