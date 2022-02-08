// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function = &Sub{}

// Sub is an element-wise subtraction function over two values.
type Sub struct {
	x1 Operand
	x2 Operand
}

// NewSub returns a new Sub Function.
func NewSub(x1, x2 Operand) *Sub {
	return &Sub{x1: x1, x2: x2}
}

// Forward computes the output of the node.
func (r *Sub) Forward() mat.Matrix[mat.Float] {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(x1v.SameDims(x2v) || x1v.VectorOfSameSize(x2v)) {
		panic("fn: matrices with not compatible size")
	}
	return x1v.Sub(x2v)
}

// Backward computes the backward pass.
func (r *Sub) Backward(gy mat.Matrix[mat.Float]) {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !(x1v.SameDims(gy) || x1v.VectorOfSameSize(gy)) &&
		!(x2v.SameDims(gy) || x2v.VectorOfSameSize(gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy)
	}
	if r.x2.RequiresGrad() {
		gx := gy.ProdScalar(-1.0)
		defer mat.ReleaseMatrix(gx)
		r.x2.PropagateGrad(gx)
	}
}
