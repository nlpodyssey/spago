// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &ELU[float32]{}

// ELU is an operator to perform the ELU activation function.
// ELU(x) = max(0,x) + min(0,α ∗ (exp(x) − 1))
type ELU[T mat.DType] struct {
	x     Operand[T]
	alpha Operand[T] // scalar
}

// NewELU returns a new ELU Function.
func NewELU[T mat.DType](x, alpha Operand[T]) *ELU[T] {
	return &ELU[T]{x: x, alpha: alpha}
}

// Forward computes the output of the function.
func (r *ELU[T]) Forward() mat.Matrix[T] {
	y := r.x.Value().ApplyWithAlpha(elu[T], r.alpha.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *ELU[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ApplyWithAlpha(eluDeriv[T], r.alpha.Value().Scalar())
		defer mat.ReleaseMatrix(gx)
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
