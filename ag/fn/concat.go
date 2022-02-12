// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

var _ Function[float32] = &Concat[float32]{}

// Concat is an operator to perform vector concatenation.
type Concat[T mat.DType] struct {
	xs    []Operand[T]
	ySize int
}

// NewConcat returns a new Concat Function.
func NewConcat[T mat.DType](xs []Operand[T]) *Concat[T] {
	return &Concat[T]{
		xs:    xs,
		ySize: 0, // assigned during the Forward()
	}
}

// Forward computes the output of the function.
func (r *Concat[T]) Forward() mat.Matrix[T] {
	r.ySize = 0 // reset output size
	ms := make([]mat.Matrix[T], len(r.xs))
	for i, x := range r.xs {
		value := x.Value()
		ms[i] = value
		r.ySize += value.Size()
	}
	return mat.ConcatV(ms...)
}

// Backward computes the backward pass.
func (r *Concat[T]) Backward(gy mat.Matrix[T]) {
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
			xs[i].PropagateGrad(gx)
		}
		mat.ReleaseMatrix(gx)
	}
}
