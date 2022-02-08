// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
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
func (r *At) Forward() mat.Matrix[mat.Float] {
	return mat.NewScalar(r.x.Value().At(r.i, r.j))
}

// Backward computes the backward pass.
func (r *At) Backward(gy mat.Matrix[mat.Float]) {
	if r.x.RequiresGrad() {
		dx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(dx)
		dx.Set(r.i, r.j, gy.Scalar())
		r.x.PropagateGrad(dx)
	}
}
