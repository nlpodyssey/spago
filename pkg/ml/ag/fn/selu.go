// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

// SeLU(x) = scale ∗ (max(0,x) + min(0, α ∗ (exp(x) − 1)))
type SeLU struct {
	x     Operand
	alpha Operand // scalar
	scale Operand // scalar
}

func NewSeLU(x, alpha, scale Operand) *SeLU {
	return &SeLU{x: x, alpha: alpha, scale: scale}
}

// Forward computes the output of the function.
func (r *SeLU) Forward() mat.Matrix {
	y := r.x.Value().ZerosLike()
	y.ApplyWithAlpha(selu, r.x.Value(), r.alpha.Value().Scalar(), r.scale.Value().Scalar())
	return y
}

func (r *SeLU) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		gx.ApplyWithAlpha(seluDeriv, r.x.Value(), r.alpha.Value().Scalar(), r.scale.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
