// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// Stack is a Function which stacks together all given operand matrices,
// producing a single bigger matrix as result.
type Stack[T mat.DType, O Operand[T]] struct {
	xs []O
}

// NewStack returns a new Stack Function.
func NewStack[T mat.DType, O Operand[T]](xs []O) *Stack[T, O] {
	return &Stack[T, O]{xs: xs}
}

// Operands returns the list of operands.
func (r *Stack[T, O]) Operands() []O {
	return r.xs
}

// Forward computes the output of the function.
func (r *Stack[T, O]) Forward() mat.Matrix[T] {
	vs := make([]mat.Matrix[T], len(r.xs))
	for i, x := range r.xs {
		vs[i] = x.Value()
	}
	return mat.Stack(vs...)
}

// Backward computes the backward pass.
func (r *Stack[T, O]) Backward(gy mat.Matrix[T]) {
	if gy.Rows() != len(r.xs) {
		panic("fn: matrices with not compatible size")
	}

	for i, x := range r.xs {
		if !x.RequiresGrad() {
			continue
		}
		gyRow := gy.ExtractRow(i).ReshapeInPlace(x.Value().Dims())
		x.AccGrad(gyRow)
		mat.ReleaseMatrix(gyRow)
	}
}
