// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"brillion.io/spago/pkg/mat"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestThresholdForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, -0.2, 3.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	threshold := &variable{
		value:        mat.NewScalar(2.0),
		grad:         nil,
		requiresGrad: false,
	}
	k := &variable{
		value:        mat.NewScalar(1.6),
		grad:         nil,
		requiresGrad: false,
	}

	f := NewThreshold(x, threshold, k)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{1.6, 1.6, 3.3, 1.6}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{0.0, 0.0, 0.8, 0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}
