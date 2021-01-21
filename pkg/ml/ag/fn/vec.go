// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &Vec{}

// Vec is a Function to reshape an matrix-operand into a column vector.
type Vec struct {
	x Operand
}

// NewVec returns a new Vec Function.
func NewVec(x Operand) *Vec {
	return &Vec{x: x}
}

// Forward computes the output of the node.
func (r *Vec) Forward() mat.Matrix {
	return r.x.Value().Reshape(r.x.Value().Size(), 1)
}

// Backward computes the backward pass.
func (r *Vec) Backward(gy mat.Matrix) {
	if !(gy.IsVector() && mat.SameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.Reshape(r.x.Value().Dims())
		defer mat.ReleaseMatrix(gx)
		r.x.PropagateGrad(gx)
	}
}
