// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// LeakyReLU is an operator to perform the LeakyReLU activation function.
// LeakyReLU(x) = max(0,x) + slope ° min(0,x)
type LeakyReLU[T mat.DType, O Operand[T]] struct {
	x     O
	alpha O // scalar
}

// NewLeakyReLU returns a new LeakyReLU Function.
func NewLeakyReLU[T mat.DType, O Operand[T]](x, alpha O) *LeakyReLU[T, O] {
	return &LeakyReLU[T, O]{x: x, alpha: alpha}
}

// Operands returns the list of operands.
func (r *LeakyReLU[T, O]) Operands() []O {
	return []O{r.x, r.alpha}
}

// Forward computes the output of the function.
func (r *LeakyReLU[T, O]) Forward() mat.Matrix[T] {
	y := r.x.Value().ApplyWithAlpha(leakyReLU[T], r.alpha.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *LeakyReLU[T, O]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(leakyReLUDeriv[T], r.alpha.Value().Scalar())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
