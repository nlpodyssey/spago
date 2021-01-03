// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &At{}

// At is an operator to obtain the i,j-th value of a matrix.
type At struct {
	x Operand
	i int
	j int
}

// NewAt returns a new At Function.
func NewAt(x Operand, i int, j int) *At {
	return &At{x: x, i: i, j: j}
}

// Forward computes the output of the function.
func (r *At) Forward() mat.Matrix {
	return mat.NewScalar(r.x.Value().At(r.i, r.j))
}

// Backward computes the backward pass.
func (r *At) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		dx := mat.NewEmptyDense(r.x.Value().Dims())
		defer mat.ReleaseDense(dx)
		dx.Set(r.i, r.j, gy.Scalar())
		r.x.PropagateGrad(dx)
	}
}
