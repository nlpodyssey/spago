// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Sqrt is an operator to perform element-wise square root function.
type Sqrt[O mat.Tensor] struct {
	x O
}

// NewSqrt returns a new Sqrt Function.
func NewSqrt[O mat.Tensor](x O) *Sqrt[O] {
	return &Sqrt[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *Sqrt[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of the function.
func (r *Sqrt[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).Sqrt(), nil
}

// Backward computes the backward pass.
func (r *Sqrt[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).Pow(-.5)
		gx.ProdScalarInPlace(.5)
		gx.ProdInPlace(gy.(mat.Matrix))
		r.x.AccGrad(gx)
	}
	return nil
}
