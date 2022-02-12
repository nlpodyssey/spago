// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

var _ Function[float32] = &ReverseSubScalar[float32]{}

// ReverseSubScalar is the element-wise subtraction function over two values.
type ReverseSubScalar[T mat.DType] struct {
	x1 Operand[T]
	x2 Operand[T] // scalar
}

// NewReverseSubScalar returns a new ReverseSubScalar Function.
func NewReverseSubScalar[T mat.DType](x1, x2 Operand[T]) *ReverseSubScalar[T] {
	return &ReverseSubScalar[T]{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *ReverseSubScalar[T]) Forward() mat.Matrix[T] {
	return mat.NewInitDense(r.x1.Value().Rows(), r.x1.Value().Columns(), r.x2.Value().Scalar()).Sub(r.x1.Value())
}

// Backward computes the backward pass.
func (r *ReverseSubScalar[T]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x1.Value(), gy) || mat.VectorsOfSameSize(r.x1.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		gx := gy.ProdScalar(-1.0)
		defer mat.ReleaseMatrix(gx)
		r.x1.PropagateGrad(gx)
	}
	if r.x2.RequiresGrad() {
		var gx T = 0.0
		for i := 0; i < gy.Rows(); i++ {
			for j := 0; j < gy.Columns(); j++ {
				gx += gy.At(i, j)
			}
		}
		scalar := mat.NewScalar(gx)
		defer mat.ReleaseDense(scalar)
		r.x2.PropagateGrad(scalar)
	}
}
