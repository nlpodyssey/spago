// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

var _ Function = &Threshold{}

// Threshold function: f(x) = x if x > threshold; k otherwise.
type Threshold struct {
	x         Operand
	threshold Operand // scalar
	k         Operand // scalar
}

// NewThreshold returns a new Threshold Function.
func NewThreshold(x, threshold, k Operand) *Threshold {
	return &Threshold{x: x, threshold: threshold, k: k}
}

// Forward computes the output of the function.
func (r *Threshold) Forward() mat.Matrix {
	y := mat.GetDenseWorkspace(r.x.Value().Dims())
	y.ApplyWithAlpha(threshold, r.x.Value(), r.threshold.Value().Scalar(), r.k.Value().Scalar())
	return y
}

// Backward computes the backward pass.
func (r *Threshold) Backward(gy mat.Matrix) {
	if !(mat.SameDims(r.x.Value(), gy) || mat.VectorsOfSameSize(r.x.Value(), gy)) {
		panic("fn: matrices with not compatible size")
	}
	if r.x.RequiresGrad() {
		gx := mat.GetDenseWorkspace(r.x.Value().Dims())
		defer mat.ReleaseDense(gx)
		gx.ApplyWithAlpha(thresholdDeriv, r.x.Value(), r.threshold.Value().Scalar(), r.k.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
