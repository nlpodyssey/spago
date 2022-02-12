// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

var _ Function[float32] = &Dot[float32]{}

// Dot is an operator to perform the dot product over two matrices.
// y = x1 dot x2
type Dot[T mat.DType] struct {
	x1 Operand[T]
	x2 Operand[T]
}

// NewDot returns a new Dot Function.
func NewDot[T mat.DType](x1, x2 Operand[T]) *Dot[T] {
	return &Dot[T]{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *Dot[T]) Forward() mat.Matrix[T] {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, x2v) || mat.VectorsOfSameSize(x1v, x2v)) {
		panic("fn: matrices with not compatible size")
	}
	var y T = 0.0
	if mat.IsVector(r.x1.Value()) && mat.IsVector(r.x2.Value()) {
		y = r.x1.Value().DotUnitary(r.x2.Value())
	} else {
		for i := 0; i < r.x1.Value().Rows(); i++ {
			for j := 0; j < r.x1.Value().Columns(); j++ {
				y += r.x1.Value().At(i, j) * r.x2.Value().At(i, j)
			}
		}
	}
	return mat.NewScalar(y)
}

// Backward computes the backward pass.
func (r *Dot[T]) Backward(gy mat.Matrix[T]) {
	if !mat.IsScalar(gy) {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x1.RequiresGrad() {
		dx := mat.GetDensePool[T]().Get(r.x1.Value().Dims())
		defer mat.ReleaseDense(dx)
		for i := 0; i < r.x1.Value().Rows(); i++ {
			for j := 0; j < r.x1.Value().Columns(); j++ {
				dx.Set(i, j, gy.Scalar()*r.x2.Value().At(i, j))
			}
		}
		r.x1.PropagateGrad(dx)
	}
	if r.x2.RequiresGrad() {
		dx := mat.GetDensePool[T]().Get(r.x2.Value().Dims())
		defer mat.ReleaseDense(dx)
		for i := 0; i < r.x2.Value().Rows(); i++ {
			for j := 0; j < r.x2.Value().Columns(); j++ {
				dx.Set(i, j, gy.Scalar()*r.x1.Value().At(i, j))
			}
		}
		r.x2.PropagateGrad(dx)
	}
}
