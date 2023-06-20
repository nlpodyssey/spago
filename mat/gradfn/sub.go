// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Sub is an element-wise subtraction function over two values.
type Sub[O mat.Tensor] struct {
	x1 O
	x2 O
}

// NewSub returns a new Sub Function.
func NewSub[O mat.Tensor](x1 O, x2 O) *Sub[O] {
	return &Sub[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *Sub[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x1, r.x2}
}

// Forward computes the output of the node.
func (r *Sub[O]) Forward() (mat.Tensor, error) {
	return r.x1.Value().(mat.Matrix).Sub(r.x2.Value().(mat.Matrix)), nil
}

// Backward computes the backward pass.
func (r *Sub[O]) Backward(gy mat.Tensor) error {
	x1v := r.x1.Value()
	x2v := r.x2.Value()
	if !mat.SameDims(x1v, gy) || !mat.SameDims(x2v, gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x1.RequiresGrad() {
		r.x1.AccGrad(gy)
	}
	if r.x2.RequiresGrad() {
		gx := gy.(mat.Matrix).ProdScalar(-1.0)
		r.x2.AccGrad(gx)
	}
	return nil
}
