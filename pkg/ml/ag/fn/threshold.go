// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"brillion.io/spago/pkg/mat"
)

// Threshold(x) = x if x > threshold; k otherwise ​
type Threshold struct {
	x         Operand
	threshold Operand // scalar
	k         Operand // scalar
}

func NewThreshold(x, threshold, k Operand) *Threshold {
	return &Threshold{x: x, threshold: threshold, k: k}
}

// Forward computes the output of the function.
func (r *Threshold) Forward() mat.Matrix {
	y := r.x.Value().ZerosLike()
	y.ApplyWithAlpha(threshold, r.x.Value(), r.threshold.Value().Scalar(), r.k.Value().Scalar())
	return y
}

func (r *Threshold) Backward(gy mat.Matrix) {
	if r.x.RequiresGrad() {
		gx := r.x.Value().ZerosLike()
		gx.ApplyWithAlpha(thresholdDeriv, r.x.Value(), r.threshold.Value().Scalar(), r.k.Value().Scalar())
		gx.ProdInPlace(gy)
		r.x.PropagateGrad(gx)
	}
}
