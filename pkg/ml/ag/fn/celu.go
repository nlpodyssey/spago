// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
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
func (r *CELU) Forward() mat.Matrix[mat.Float] {
	y := mat.GetDensePool[mat.Float]().Get(r.x.Value().Dims())
	y.ApplyWithAlpha(celu, r.x.Value(), r.alpha.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *CELU) Backward(gy mat.Matrix[mat.Float]) {
	if !(r.x.Value().SameDims(gy) || r.x.Value().VectorOfSameSize(gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDensePool[mat.Float]().Get(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.ApplyWithAlpha(celuDeriv, r.x.Value(), r.alpha.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
