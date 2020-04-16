// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

// ELU(x) = max(0,x) + min(0,α ∗ (exp(x) − 1))
type ELU struct {
	x     Operand
	alpha Operand // scalar
}

func NewELU(x, alpha Operand) *ELU {
	return &ELU{x: x, alpha: alpha}
}

// Forward computes the output of the function.
func (r *ELU) Forward() mat.Matrix {
	y := r.x.Value().ZerosLike()
	y.ApplyWithAlpha(elu, r.x.Value(), r.alpha.Value().Scalar())
	return y
}

func (r *ELU) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		gx.ApplyWithAlpha(eluDeriv, r.x.Value(), r.alpha.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
