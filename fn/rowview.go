// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// RowView is a function to extract the i-th row from the input matrix.
type RowView[O DualValue] struct {
	x O
	i int
}

// NewRowView returns a new RowView Function.
func NewRowView[O DualValue](x O, i int) *RowView[O] {
	if i < 0 {
		panic("fn: invalid row index")
	}
	return &RowView[O]{
		x: x,
		i: i,
	}
}

// Operands returns the list of operands.
func (r *RowView[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *RowView[O]) Forward() (mat.Matrix, error) {
	return r.x.Value().ExtractRow(r.i), nil
}

// Backward computes the backward pass.
func (r *RowView[O]) Backward(gy mat.Matrix) error {
	if !(r.x.Value().Cols() == gy.Size()) {
		return fmt.Errorf("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		for j := 0; j < r.x.Value().Cols(); j++ {
			gx.SetScalar(gy.ScalarAt(0, j), r.i, j)
		}
		r.x.AccGrad(gx)
	}
	return nil
}
