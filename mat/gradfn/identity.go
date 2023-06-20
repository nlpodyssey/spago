// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Copy is an operator to perform copy function.
// y = x
type Copy[O mat.Tensor] struct {
	x O
}

// NewCopy returns a new Copy Function.
func NewCopy[O mat.Tensor](x O) *Copy[O] {
	return &Copy[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *Copy[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of the function.
func (r *Copy[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).Clone(), nil
}

// Backward computes the backward pass.
func (r *Copy[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	r.x.AccGrad(gy)
	return nil
}
