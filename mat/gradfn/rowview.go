// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// RowView is a function to extract the i-th row from the input matrix.
type RowView[O mat.Tensor] struct {
	x O
	i int
}

// NewRowView returns a new RowView Function.
func NewRowView[O mat.Tensor](x O, i int) *RowView[O] {
	if i < 0 {
		panic("fn: invalid row index")
	}
	return &RowView[O]{
		x: x,
		i: i,
	}
}

// Operands returns the list of operands.
func (r *RowView[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of the function.
func (r *RowView[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).ExtractRow(r.i), nil
}

// Backward computes the backward pass.
func (r *RowView[O]) Backward(gy mat.Tensor) error {
	if !(r.x.Value().Shape()[1] == gy.Size()) {
		return fmt.Errorf("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).ZerosLike()
		for j := 0; j < r.x.Value().Shape()[1]; j++ {
			gx.SetScalar(gy.(mat.Matrix).ScalarAt(0, j), r.i, j)
		}
		r.x.AccGrad(gx)
	}
	return nil
}
