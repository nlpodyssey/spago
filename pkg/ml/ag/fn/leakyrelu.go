// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"brillion.io/spago/pkg/mat"
)

// LeakyReLU(x) = max(0,x) + slope ° min(0,x)
type LeakyReLU struct {
	x     Operand
	alpha Operand // scalar
}

func NewLeakyReLU(x, alpha Operand) *LeakyReLU {
	return &LeakyReLU{x: x, alpha: alpha}
}

// Forward computes the output of the function.
func (r *LeakyReLU) Forward() mat.Matrix {
	y := r.x.Value().ZerosLike()
	y.ApplyWithAlpha(leakyReLU, r.x.Value(), r.alpha.Value().Scalar())
	return y
}

func (r *LeakyReLU) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		gx.ApplyWithAlpha(leakyReLUDeriv, r.x.Value(), r.alpha.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
