// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// Min is an operator to perform element-wise min.
// y = min(x1, x2)
type Min[T mat.DType, O Operand[T]] struct {
	x1       O
	x2       O
	operands []O
}

// NewMin returns a new Min Function.
func NewMin[T mat.DType, O Operand[T]](x1 O, x2 O) *Min[T, O] {
	return &Min[T, O]{
		x1:       x1,
		x2:       x2,
		operands: []O{x1, x2},
	}
}

// Operands returns the list of operands.
func (r *Min[T, O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *Min[T, O]) Forward() mat.Matrix[T] {
	return r.x1.Value().Minimum(r.x2.Value())
}

// Backward computes the backward pass.
func (r *Min[T, O]) Backward(gy mat.Matrix[T]) {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(mat.SameDims(x1v, gy) || mat.VectorsOfSameSize(x1v, gy)) &&
		!(mat.SameDims(x2v, gy) || mat.VectorsOfSameSize(x2v, gy)) {
		panic("fn: matrices with not compatible size")
	}

	n := gy.Size()
	gyData := mat.Data[T](gy)
	x1vData := mat.Data[T](x1v)
	x2vData := mat.Data[T](x2v)

	if r.x1.RequiresGrad() {
		gx := x1v.ZerosLike()
		defer mat.ReleaseMatrix(gx)
		gxData := mat.Data[T](gx)
		for i := 0; i < n; i++ {
			if x1vData[i] < x2vData[i] {
				gxData[i] = gyData[i]
			}
		}
		r.x1.AccGrad(gx)
	}
	if r.x2.RequiresGrad() {
		gx := x2v.ZerosLike()
		defer mat.ReleaseMatrix(gx)
		n := gy.Size()
		gxData := mat.Data[T](gx)
		for i := 0; i < n; i++ {
			if x2vData[i] < x1vData[i] {
				gxData[i] = gyData[i]
			}
		}
		r.x2.AccGrad(gx)
	}
}
