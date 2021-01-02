// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import mat "github.com/nlpodyssey/spago/pkg/mat32"

var _ Function = &RotateR{}

// RotateR is a function to perform a right circular shift of a vector.
type RotateR struct {
	x Operand
	i int
}

// NewRotateR returns a new RotateR Function. `i` is the number of places by
// which the elements are shifted.
func NewRotateR(x Operand, i int) *RotateR {
	return &RotateR{x: x, i: i}
}

// Forward computes the output of the function.
func (r *RotateR) Forward() mat.Matrix {
	xv := r.x.Value().Data()
	return mat.NewVecDense(rotateR(xv, r.i))
}

// Backward computes the backward pass.
func (r *RotateR) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		gx := mat.NewVecDense(rotateL(gy.Data(), r.i))
		defer mat.ReleaseDense(gx)
		r.x.PropagateGrad(gx)
	}
}

func rotateR(a []mat.Float, i int) []mat.Float {
	x, b := a[:(len(a)-i)], a[(len(a)-i):]
	return append(b, x...)
}

func rotateL(a []mat.Float, i int) []mat.Float {
	x, b := a[:i], a[i:]
	return append(b, x...)
}
