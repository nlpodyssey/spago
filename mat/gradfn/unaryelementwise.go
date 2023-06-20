// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// UnaryElementwise is a single-input element-wise function.
type UnaryElementwise[O mat.Tensor] struct {
	x  O
	f  func(i, j int, v float64) float64 // function
	df func(i, j int, v float64) float64 // derivative
}

// Operands returns the list of operands.
func (r *UnaryElementwise[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of this node.
func (r *UnaryElementwise[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).Apply(r.f), nil
}

// Backward computes the backward pass.
func (r *UnaryElementwise[O]) Backward(gy mat.Tensor) error {
	if !mat.SameDims(r.x.Value(), gy) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if r.x.RequiresGrad() {
		gx := r.x.Value().(mat.Matrix).Apply(r.df)
		gx.ProdInPlace(gy.(mat.Matrix))
		r.x.AccGrad(gx)
	}
	return nil
}
