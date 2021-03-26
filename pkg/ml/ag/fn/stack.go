// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import mat "github.com/nlpodyssey/spago/pkg/mat32"

var _ Function = &Stack{}

// Stack is a Function which stacks together all given operand matrices,
// producing a single bigger matrix as result.
type Stack struct {
	xs []Operand
}

// NewStack returns a new Stack Function.
func NewStack(xs []Operand) *Stack {
	return &Stack{xs: xs}
}

// Forward computes the output of the function.
func (r *Stack) Forward() mat.Matrix {
	vs := make([]mat.Matrix, len(r.xs))
	for i, x := range r.xs {
		vs[i] = x.Value()
	}
	return mat.Stack(vs...)
}

// Backward computes the backward pass.
func (r *Stack) Backward(gy mat.Matrix) {
	if gy.Rows() != len(r.xs) {
		panic("fn: matrices with not compatible size")
	}
	sizes := make([]int, len(r.xs))
	for i, x := range r.xs {
		sizes[i] = x.Value().Size()
		if !(sizes[i] == gy.Columns()) {
			panic("fn: matrices with not compatible size")
		}
	}
	xs := r.xs
	for i, gx := range gy.SplitV(sizes...) {
		if xs[i].RequiresGrad() {
			xs[i].PropagateGrad(gx)
		}
		mat.ReleaseMatrix(gx)
	}
}
