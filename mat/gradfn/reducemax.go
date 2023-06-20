// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// ReduceMax is an operator to perform reduce-max function.
// It gets the maximum element of the Operand x
type ReduceMax[O mat.Tensor] struct {
	x      O
	argmax int
}

// NewReduceMax returns a new ReduceMax Function.
func NewReduceMax[O mat.Tensor](x O) *ReduceMax[O] {
	return &ReduceMax[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *ReduceMax[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of this function.
func (r *ReduceMax[O]) Forward() (mat.Tensor, error) {
	xv := r.x.Value()
	r.argmax = xv.(mat.Matrix).ArgMax()
	return xv.(mat.Matrix).At(r.argmax), nil
}

// Backward computes the backward pass.
func (r *ReduceMax[O]) Backward(gy mat.Tensor) error {
	if !mat.IsScalar(gy) {
		return fmt.Errorf("fn: the gradient had to be a scalar")
	}
	if r.x.RequiresGrad() {
		x := r.x.Value()
		gx := x.(mat.Matrix).ZerosLike()
		gx.SetAt(gy.(mat.Matrix), r.argmax)
		r.x.AccGrad(gx)
	}
	return nil
}
