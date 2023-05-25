// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// At is an operator to obtain the i,j-th value of a matrix.
type At[O DualValue] struct {
	x       O
	indices []int
}

// NewAt returns a new At Function.
func NewAt[O DualValue](x O, indices ...int) *At[O] {
	return &At[O]{
		x:       x,
		indices: indices,
	}
}

// Operands returns the list of operands.
func (r *At[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *At[O]) Forward() (mat.Matrix, error) {
	return r.x.Value().At(r.indices...), nil
}

// Backward computes the backward pass.
func (r *At[O]) Backward(gy mat.Matrix) error {
	if r.x.RequiresGrad() {
		dx := r.x.Value().ZerosLike()
		dx.SetAt(gy, r.indices...)
		r.x.AccGrad(dx)
	}
	return nil
}
