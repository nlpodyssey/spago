// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Softmax is a single-input softmax function.
type Softmax[T mat.DType, O Operand[T]] struct {
	x O
	y mat.Matrix[T] // initialized during the forward pass (required by the backward pass)
}

// NewSoftmax returns a new Softmax Function.
func NewSoftmax[T mat.DType, O Operand[T]](x O) *Softmax[T, O] {
	return &Softmax[T, O]{x: x}
}

// Operands returns the list of operands.
func (r *Softmax[T, O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of this function.
func (r *Softmax[T, O]) Forward() mat.Matrix[T] {
	r.y = r.x.Value().Softmax()
	return r.y
}

// Backward computes the backward pass.
func (r *Softmax[T, O]) Backward(gy mat.Matrix[T]) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		y := r.y
		n := y.Size()
		jb := mat.NewInitFuncDense[T](n, n, func(row, col int) T {
			if row == col {
				v := y.AtVec(row)
				return v * (1 - v)
			}
			return -(y.AtVec(row) * y.AtVec(col))
		})
		defer mat.ReleaseDense(jb)
		gx := jb.Mul(gy)
		defer mat.ReleaseMatrix(gx)
		r.x.PropagateGrad(gx)
	}
}
