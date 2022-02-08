// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

var _ Function = &AtVec{}

// AtVec is an operator to obtain the i-th value of a vector.
type AtVec struct {
	x Operand
	i int
}

// NewAtVec returns a new AtVec Function.
func NewAtVec(x Operand, i int) *AtVec {
	return &AtVec{x: x, i: i}
}

// Forward computes the output of the function.
func (r *AtVec) Forward() mat.Matrix[mat.Float] {
	return mat.NewScalar(r.x.Value().AtVec(r.i))
}

// Backward computes the backward pass.
func (r *AtVec) Backward(gy mat.Matrix[mat.Float]) {
	if r.x.RequiresGrad() {
		dx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(dx)
		dx.SetVec(r.i, gy.Scalar())
		r.x.PropagateGrad(dx)
	}
}
