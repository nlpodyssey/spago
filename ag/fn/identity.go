// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/mat"

// Identity is an operator to perform identity function.
// y = x
type Identity[O Operand] struct {
	x        O
	operands []O
}

// NewIdentity returns a new Identity Function.
func NewIdentity[O Operand](x O) *Identity[O] {
	return &Identity[O]{
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *Identity[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *Identity[O]) Forward() mat.Matrix {
	return r.x.Value().Clone()
}

// Backward computes the backward pass.
func (r *Identity[O]) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	r.x.AccGrad(gy)
}
