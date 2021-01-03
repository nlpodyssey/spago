// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &LeakyReLU{}

// LeakyReLU is an operator to perform the LeakyReLU activation function.
// LeakyReLU(x) = max(0,x) + slope ° min(0,x)
type LeakyReLU struct {
	x     Operand
	alpha Operand // scalar
}

// NewLeakyReLU returns a new LeakyReLU Function.
func NewLeakyReLU(x, alpha Operand) *LeakyReLU {
	return &LeakyReLU{x: x, alpha: alpha}
}

// Forward computes the output of the function.
func (r *LeakyReLU) Forward() mat.Matrix {
	y := mat.GetDenseWorkspace(r.x.Value().Dims())
	y.ApplyWithAlpha(leakyReLU, r.x.Value(), r.alpha.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *LeakyReLU) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDenseWorkspace(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.ApplyWithAlpha(leakyReLUDeriv, r.x.Value(), r.alpha.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
