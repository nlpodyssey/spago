// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Dot is an operator to perform the dot product over two matrices.
// y = x1 dot x2
type Dot[O mat.Tensor] struct {
	x1 O
	x2 O
}

// NewDot returns a new Dot Function.
func NewDot[O mat.Tensor](x1 O, x2 O) *Dot[O] {
	return &Dot[O]{
		x1: x1,
		x2: x2,
	}
}

// Operands returns the list of operands.
func (r *Dot[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x1, r.x2}
}

// Forward computes the output of the function.
func (r *Dot[O]) Forward() (mat.Tensor, error) {
	x1v := r.x1.Value().(mat.Matrix)
	x2v := r.x2.Value().(mat.Matrix)
	if !mat.SameDims(x1v, x2v) {
		return nil, fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if mat.IsVector(x1v) && mat.IsVector(x2v) {
		return x1v.DotUnitary(x2v), nil
	}
	prod := x1v.Prod(x2v)
	return prod.Sum(), nil
}

// Backward computes the backward pass.
func (r *Dot[O]) Backward(gy mat.Tensor) error {
	if !mat.IsScalar(gy.(mat.Matrix)) {
		return fmt.Errorf("fn: the gradient had to be a scalar")
	}
	gys := gy.Item().F64()
	if r.x1.RequiresGrad() {
		gx := r.x2.Value().(mat.Matrix).ProdScalar(gys)
		r.x1.AccGrad(gx)
	}
	if r.x2.RequiresGrad() {
		gx := r.x1.Value().(mat.Matrix).ProdScalar(gys)
		r.x2.AccGrad(gx)
	}
	return nil
}
