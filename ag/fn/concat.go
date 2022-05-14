// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Concat is an operator to perform vector concatenation.
type Concat[T mat.DType, O Operand[T]] struct {
	xs    []O
	ySize int
}

// NewConcat returns a new Concat Function.
func NewConcat[T mat.DType, O Operand[T]](xs []O) *Concat[T, O] {
	return &Concat[T, O]{
		xs:    xs,
		ySize: 0, // assigned during the Forward()
	}
}

// Operands returns the list of operands.
func (r *Concat[T, O]) Operands() []O {
	return r.xs
}

// Forward computes the output of the function.
func (r *Concat[T, O]) Forward() mat.Matrix {
	r.ySize = 0 // reset output size
	ms := make([]mat.Matrix, len(r.xs))
	for i, x := range r.xs {
		value := x.Value()
		ms[i] = value
		r.ySize += value.Size()
	}
	return mat.ConcatV[T](ms...)
}

// Backward computes the backward pass.
func (r *Concat[T, O]) Backward(gy mat.Matrix) {
	if r.ySize != gy.Size() {
		panic("fn: vectors with not compatible size")
	}
	sizes := make([]int, len(r.xs))
	for i, x := range r.xs {
		sizes[i] = x.Value().Size()
	}
	xs := r.xs
	for i, gx := range gy.SplitV(sizes...) {
		if xs[i].RequiresGrad() {
			xs[i].AccGrad(gx)
		}
		mat.ReleaseMatrix(gx)
	}
}
