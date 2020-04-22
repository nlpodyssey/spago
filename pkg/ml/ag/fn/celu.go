// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

// CeLU(x) = max(0,x) + min(0,α ∗ (exp(x/α) − 1))
type CeLU struct {
	x     Operand
	alpha Operand // scalar
}

func NewCeLU(x, alpha Operand) *CeLU {
	return &CeLU{x: x, alpha: alpha}
}

// Forward computes the output of the function.
func (r *CeLU) Forward() mat.Matrix {
	y := r.x.Value().ZerosLike()
	y.ApplyWithAlpha(celu, r.x.Value(), r.alpha.Value().Scalar())
	return y
}

func (r *CeLU) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		defer mat.ReleaseDense(gx.(*mat.Dense))
		gx.ApplyWithAlpha(celuDeriv, r.x.Value(), r.alpha.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
