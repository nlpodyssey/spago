// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Flatten is a Function to reshape a matrix-operand into a "flattened" row vector.
type Flatten[O Operand] struct {
	x O
}

// NewFlatten returns a new Flatten Function.
func NewFlatten[O Operand](x O) *Flatten[O] {
	return &Flatten[O]{
		x: x,
	}
}

// Operands returns the list of operands.
func (r *Flatten[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the node.
func (r *Flatten[O]) Forward() mat.Matrix {
	return r.x.Value().Flatten()
}

// Backward computes the backward pass.
func (r *Flatten[O]) Backward(gy mat.Matrix) {
	if !(mat.IsVector(gy) && r.x.Value().Size() == gy.Size()) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := gy.Reshape(r.x.Value().Dims())
		defer mat.ReleaseMatrix(gx)
		r.x.AccGrad(gx)
	}
}
