// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function = &SubScalar{}

// SubScalar is an element-wise subtraction function with a scalar value.
type SubScalar struct {
	x1 Operand
	x2 Operand // scalar
}

// NewSubScalar returns a new SubScalar Function.
func NewSubScalar(x1, x2 Operand) *SubScalar {
	return &SubScalar{x1: x1, x2: x2}
}

// Forward computes the output of the node.
func (r *SubScalar) Forward() mat.Matrix[mat.Float] {
	return r.x1.Value().SubScalar(r.x2.Value().Scalar())
}

// Backward computes the backward pass.
func (r *SubScalar) Backward(gy mat.Matrix[mat.Float]) {
	if !(r.x1.Value().SameDims(gy) || r.x1.Value().VectorOfSameSize(gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x1.RequiresGrad() {
		r.x1.PropagateGrad(gy) // equals to gy.ProdScalar(1.0)
	}
	if r.x2.RequiresGrad() {
		var gx mat.Float = 0.0
		for i := 0; i < gy.Rows(); i++ {
			for j := 0; j < gy.Columns(); j++ {
				gx -= gy.At(i, j)
			}
		}
		scalar := mat.NewScalar(gx)
		defer mat.ReleaseDense(scalar)
		r.x2.PropagateGrad(scalar)
	}
}
