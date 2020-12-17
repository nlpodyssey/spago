// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import "github.com/nlpodyssey/spago/pkg/mat"

var _ Function = &RotateR{}

type RotateR struct {
	x Operand
	i int
}

func NewRotateR(x Operand, i int) *RotateR {
	return &RotateR{x: x, i: i}
}

// Forward computes the output of the function.
func (r *RotateR) Forward() mat.Matrix {
	xv := r.x.Value().Data()
	return mat.NewVecDense(rotateR(xv, r.i))
}

func (r *RotateR) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		gx := mat.NewVecDense(rotateL(gy.Data(), r.i))
		defer mat.ReleaseDense(gx)
		r.x.PropagateGrad(gx)
	}
}

func rotateR(a []float64, i int) []float64 {
	x, b := a[:(len(a)-i)], a[(len(a)-i):]
	return append(b, x...)
}

func rotateL(a []float64, i int) []float64 {
	x, b := a[:i], a[i:]
	return append(b, x...)
}
