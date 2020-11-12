// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/pkg/mat"

var _ Function = &Stack{}

type Stack struct {
	xs []Operand
}

func NewStack(xs []Operand) *Stack {
	return &Stack{xs: xs}
}

// Forward computes the output of the function.
func (r *Stack) Forward() mat.Matrix {
	rows := len(r.xs)
	cols := r.xs[0].Value().Rows()
	ms := mat.GetDenseWorkspace(rows, cols) // it doesn't need to be empty, because we are going to fill it up again
	for i, x := range r.xs {
		row := x.Value()
		for j := 0; j < cols; j++ {
			ms.Set(i, j, row.At(j, 0))
		}
	}
	return ms
}

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
	for i, gx := range gy.(*mat.Dense).SplitV(sizes...) {
		if xs[i].RequiresGrad() {
			xs[i].PropagateGrad(gx)
		}
		mat.ReleaseDense(gx.(*mat.Dense))
	}
}
