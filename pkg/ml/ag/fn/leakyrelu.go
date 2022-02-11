// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function[float32] = &LeakyReLU[float32]{}

// LeakyReLU is an operator to perform the LeakyReLU activation function.
// LeakyReLU(x) = max(0,x) + slope ° min(0,x)
type LeakyReLU[T mat.DType] struct {
	x     Operand[T]
	alpha Operand[T] // scalar
}

// NewLeakyReLU returns a new LeakyReLU Function.
func NewLeakyReLU[T mat.DType](x, alpha Operand[T]) *LeakyReLU[T] {
	return &LeakyReLU[T]{x: x, alpha: alpha}
}

// Forward computes the output of the function.
func (r *LeakyReLU[T]) Forward() mat.Matrix[T] {
	y := mat.GetDensePool[T]().Get(r.x.Value().Dims())
	y.ApplyWithAlpha(leakyReLU[T], r.x.Value(), r.alpha.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *LeakyReLU[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDensePool[T]().Get(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.ApplyWithAlpha(leakyReLUDeriv[T], r.x.Value(), r.alpha.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
