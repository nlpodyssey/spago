// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"gonum.org/v1/gonum/floats"
	"saientist.dev/spago/pkg/mat"
	"testing"
)

func TestSoftPlusForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, -0.2, 20.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	beta := &variable{
		value:        mat.NewScalar(2.0),
		grad:         nil,
		requiresGrad: false,
	}
	threshold := &variable{
		value:        mat.NewScalar(20.0),
		grad:         nil,
		requiresGrad: false,
	}

	f := NewSoftPlus(x, beta, threshold)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.399069434, 0.25650762, 20.3, 0.346573590}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-0.5498339, 0.20065616, 0.8, 0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}
