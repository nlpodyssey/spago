// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// RotateR is a function to perform a right circular shift of a vector.
type RotateR[O Operand] struct {
	x O
	i int
}

// NewRotateR returns a new RotateR Function. `i` is the number of places by
// which the elements are shifted.
func NewRotateR[O Operand](x O, i int) *RotateR[O] {
	return &RotateR[O]{
		x: x,
		i: i,
	}
}

// Operands returns the list of operands.
func (r *RotateR[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *RotateR[O]) Forward() mat.Matrix {
	x := r.x.Value()
	return rotate(x, x.Size()-r.i)
}

// Backward computes the backward pass.
func (r *RotateR[O]) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		gx := rotate(gy, r.i)
		defer mat.ReleaseMatrix(gx)
		r.x.AccGrad(gx)
	}
}

func rotate(m mat.Matrix, i int) mat.Matrix {
	size := m.Size()

	left := m.Slice(0, 0, i, 1)
	defer mat.ReleaseMatrix(left)

	right := m.Slice(i, 0, size, 1)
	defer mat.ReleaseMatrix(right)

	return m.NewConcatV(right, left)
}
