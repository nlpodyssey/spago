// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Reshape is a Function which reshapes an operand into a new matrix of given
// rows × columns size.
type Reshape[O mat.Tensor] struct {
	x    O
	rows int
	cols int
}

// NewReshape returns a new Reshape Function.
func NewReshape[O mat.Tensor](x O, r, c int) *Reshape[O] {
	return &Reshape[O]{
		x:    x,
		rows: r,
		cols: c,
	}
}

// Operands returns the list of operands.
func (r *Reshape[O]) Operands() []mat.Tensor {
	return []mat.Tensor{r.x}
}

// Forward computes the output of the node.
func (r *Reshape[O]) Forward() (mat.Tensor, error) {
	return r.x.Value().(mat.Matrix).Reshape(r.rows, r.cols), nil
}

// Backward computes the backward pass.
func (r *Reshape[O]) Backward(gy mat.Tensor) error {
	if gy.Shape()[1] != r.cols && gy.Shape()[0] != r.rows {
		return fmt.Errorf("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.(mat.Matrix).Reshape(r.x.Value().Shape()...)
		r.x.AccGrad(gx)
	}
	return nil
}
