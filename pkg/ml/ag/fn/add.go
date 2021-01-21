// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import mat "github.com/nlpodyssey/spago/pkg/mat32"

var _ Function = &Add{}

// Add is an operator to perform element-wise sum over two values.
// y = x1 + x2
type Add struct {
	x1 Operand
	x2 Operand
}

// NewAdd returns a new Add Function.
func NewAdd(x1, x2 Operand) *Add {
	return &Add{x1: x1, x2: x2}
}

// Forward computes the output of the function.
func (r *Add) Forward() mat.Matrix {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if x1v == nil {
		x1v = x2v.ZerosLike()
		defer mat.ReleaseMatrix(x1v)
	}
	if !(mat.SameDims(x1v, x2v) || mat.VectorsOfSameSize(x1v, x2v)) {
		panic("fn: matrices with not compatible size")
	}
	return x1v.Add(x2v)
}

// Backward computes the backward pass.
func (r *Add) Backward(gy mat.Matrix) {
	if r.x1.RequiresGrad() {
		x1v := r.x1.Value()
		if !(mat.SameDims(x1v, gy) || mat.VectorsOfSameSize(x1v, gy)) {
			panic("fn: matrices with not compatible size")
		}
		r.x1.PropagateGrad(gy)
	}
	if r.x2.RequiresGrad() {
		x2v := r.x2.Value()
		if !(mat.SameDims(x2v, gy) || mat.VectorsOfSameSize(x2v, gy)) {
			panic("fn: matrices with not compatible size")
		}
		r.x2.PropagateGrad(gy)
	}
}
