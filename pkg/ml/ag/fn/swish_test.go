// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"gonum.org/v1/gonum/floats"
	"saientist.dev/spago/pkg/mat"
	"testing"
)

func TestSwishForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	beta := &variable{
		value:        mat.NewScalar(2.0),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSwish(x, beta)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.0549833997, -0.080262468, 0.1936968919, 0.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-0.5993373119, 0.1526040208, 0.6263414804, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}

	if !floats.EqualApprox(beta.grad.Data(), []float64{0.0188025145}, 1.0e-6) {
		t.Error("The beta-gradients don't match the expected values")
	}
}
