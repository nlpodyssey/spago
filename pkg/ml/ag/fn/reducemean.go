// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &ReduceMean{}

// ReduceMean is an operator to perform reduce-mean function.
type ReduceMean struct {
	x Operand
}

// NewReduceMean returns a new ReduceMean Function.
func NewReduceMean(x Operand) *ReduceMean {
	return &ReduceMean{x: x}
}

// Forward computes the output of this node.
func (r *ReduceMean) Forward() mat.Matrix {
	return mat.NewScalar(r.x.Value().Sum() / mat.Float(r.x.Value().Size()))
}

// Backward computes the backward pass.
func (r *ReduceMean) Backward(gy mat.Matrix) {
	if !gy.IsScalar() {
		panic("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		gx := mat.NewInitVecDense(r.x.Value().Size(), gy.Scalar()/mat.Float(r.x.Value().Size()))
		defer mat.ReleaseDense(gx)
		r.x.PropagateGrad(gx)
	}
}
