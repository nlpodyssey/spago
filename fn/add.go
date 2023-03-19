// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// Add is an operator to perform element-wise sum over two values.
// y = x1 + x2
type Add[O DualValue] struct {
	x1 O
	x2 O
}

// NewAdd returns a new Add Function.
func NewAdd[O DualValue](x1, x2 O) *Add[O] {
	return &Add[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *Add[O]) Operands() []O {
	return []O{r.x1, r.x2}
}

// Forward computes the output of the function.
func (r *Add[O]) Forward() mat.Matrix {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if x1v == nil {
		x1v = x2v.ZerosLike()
		defer mat.ReleaseMatrix(x1v)
	}
	return x1v.Add(x2v)
}

// Backward computes the backward pass.
func (r *Add[O]) Backward(gy mat.Matrix) {
	if r.x1.RequiresGrad() {
		x1v := r.x1.Value()
		if !mat.SameDims(x1v, gy) {
			panic("fn: matrices have incompatible dimensions")
		}
		r.x1.AccGrad(gy)
	}
	if r.x2.RequiresGrad() {
		x2v := r.x2.Value()
		if !mat.SameDims(x2v, gy) {
			panic("fn: matrices have incompatible dimensions")
		}
		r.x2.AccGrad(gy)
	}
}
