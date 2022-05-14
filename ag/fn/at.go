// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// At is an operator to obtain the i,j-th value of a matrix.
type At[O Operand] struct {
	x        O
	i        int
	j        int
	operands []O
}

// NewAt returns a new At Function.
func NewAt[O Operand](x O, i int, j int) *At[O] {
	return &At[O]{
		x:        x,
		i:        i,
		j:        j,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (r *At[O]) Operands() []O {
	return r.operands
}

// Forward computes the output of the function.
func (r *At[O]) Forward() mat.Matrix {
	return r.x.Value().At(r.i, r.j)
}

// Backward computes the backward pass.
func (r *At[O]) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		dx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(dx)
		dx.Set(r.i, r.j, gy)
		r.x.AccGrad(dx)
	}
}
