// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"github.com/nlpodyssey/spago/mat"
)

// At is an operator to obtain the i,j-th value of a matrix.
type At[O mat.Tensor] struct {
	x       O
	indices []int
}

// NewAt returns a new At Function.
func NewAt[O mat.Tensor](x O, indices ...int) *At[O] {
	return &At[O]{
		x:       x,
		indices: indices,
	}
}

// Operands returns the list of operands.
func (r *At[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of the function.
func (r *At[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).At(r.indices...), nil
}

// Backward computes the backward pass.
func (r *At[O]) Backward(gy mat.Tensor) error {
	if r.x.RequiresGrad() {
		dx := r.x.Value().(mat.Matrix).ZerosLike()
		dx.SetAt(gy.(mat.Matrix), r.indices...)
		r.x.AccGrad(dx)
	}
	return nil
}
