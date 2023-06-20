// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// ColView is an operator to extract the i-th column from a matrix.
type ColView[O mat.Tensor] struct {
	x O
	i int
}

// NewColView extracts the i-th column from the input matrix.
func NewColView[O mat.Tensor](x O, i int) *ColView[O] {
	if i < 0 {
		panic("fn: invalid column index")
	}
	return &ColView[O]{
		x: x,
		i: i,
	}
}

// Operands returns the list of operands.
func (r *ColView[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of the function.
func (r *ColView[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).ExtractColumn(r.i), nil
}

// Backward computes the backward pass.
func (r *ColView[O]) Backward(gy mat.Tensor) error {
	if !(r.x.Value().Shape()[0] == gy.Size()) {
		return fmt.Errorf("fn: the number of rows of the input matrix must be equal to the number of rows of the gradient")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).ZerosLike()
		for i := 0; i < r.x.Value().Shape()[0]; i++ {
			gx.SetScalar(gy.(mat.Matrix).ScalarAt(i), i, r.i)
		}
		r.x.AccGrad(gx)
	}
	return nil
}
