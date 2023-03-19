// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// AtVec is an operator to obtain the i-th value of a vector.
type AtVec[O DualValue] struct {
	x O
	i int
}

// NewAtVec returns a new AtVec Function.
func NewAtVec[O DualValue](x O, i int) *AtVec[O] {
	return &AtVec[O]{
		x: x,
		i: i,
	}
}

// Operands returns the list of operands.
func (r *AtVec[O]) Operands() []O {
	return []O{r.x}
}

// Forward computes the output of the function.
func (r *AtVec[O]) Forward() mat.Matrix {
	return r.x.Value().AtVec(r.i)
}

// Backward computes the backward pass.
func (r *AtVec[O]) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		dx := r.x.Value().ZerosLike()
		defer mat.ReleaseMatrix(dx)
		dx.SetVec(r.i, gy)
		r.x.AccGrad(dx)
	}
}
