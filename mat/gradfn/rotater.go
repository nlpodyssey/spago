// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import "github.com/nlpodyssey/spago/mat"

// RotateR is a function to perform a right circular shift of a vector.
type RotateR[O mat.Tensor] struct {
	x O
	i int
}

// NewRotateR returns a new RotateR Function. `i` is the number of places by
// which the elements are shifted.
func NewRotateR[O mat.Tensor](x O, i int) *RotateR[O] {
	return &RotateR[O]{
		x: x,
		i: i,
	}
}

// Operands returns the list of operands.
func (r *RotateR[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of the function.
func (r *RotateR[O]) Forward() (mat.Tensor, error) {
	x := r.x.Value().(mat.Matrix)
	return rotate(x, x.Size()-r.i), nil
}

// Backward computes the backward pass.
func (r *RotateR[O]) Backward(gy mat.Tensor) error {
	if r.x.RequiresGrad() {
		gx := rotate(gy.(mat.Matrix), r.i)
		r.x.AccGrad(gx)
	}
	return nil
}

func rotate(m mat.Matrix, i int) mat.Matrix {
	size := m.Size()
	left := m.Slice(0, 0, i, 1)
	right := m.Slice(i, 0, size, 1)
	return m.NewConcatV(right, left)
}
