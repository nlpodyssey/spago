// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// ColView is an operator to extract the i-th column from a matrix.
type ColView[O DualValue] struct {
	x O
	i int
}

// NewColView extracts the i-th column from the input matrix.
func NewColView[O DualValue](x O, i int) *ColView[O] {
	if i < 0 {
		panic("fn: invalid column index")
	}
	return &ColView[O]{
		x: x,
		i: i,
	}
}

// Operands returns the list of operands.
func (r *ColView[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *ColView[O]) Forward() (mat.Matrix, error) {
	return r.x.Value().ExtractColumn(r.i), nil
}

// Backward computes the backward pass.
func (r *ColView[O]) Backward(gy mat.Matrix) error {
	if !(r.x.Value().Rows() == gy.Size()) {
		return fmt.Errorf("fn: the number of rows of the input matrix must be equal to the number of rows of the gradient")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(gx)
		for i := 0; i < r.x.Value().Rows(); i++ {
			gx.SetScalar(i, r.i, gy.ScalarAtVec(i))
		}
		r.x.AccGrad(gx)
	}
	return nil
}
