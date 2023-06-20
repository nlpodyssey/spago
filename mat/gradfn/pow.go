// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Pow is an operator to perform element-wise pow function.
type Pow[O mat.Tensor] struct {
	x     O
	power float64
}

// NewPow returns a new Pow Function.
func NewPow[O mat.Tensor](x O, power float64) *Pow[O] {
	return &Pow[O]{
		x:     x,
		power: power,
	}
}

// Operands returns the list of operands.
func (r *Pow[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of the function.
func (r *Pow[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).Pow(r.power), nil
}

// Backward computes the backward pass.
func (r *Pow[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).Pow(r.power - 1)
		gx.ProdScalarInPlace(r.power).ProdInPlace(gy.(mat.Matrix))
		r.x.AccGrad(gx)
	}
	return nil
}
