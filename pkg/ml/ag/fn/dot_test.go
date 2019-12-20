// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"brillion.io/spago/pkg/mat"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestDot_Forward(t *testing.T) {

	x1 := &variable{
		value: mat.NewDense(3, 4, []float64{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	x2 := &variable{
		value: mat.NewDense(3, 4, []float64{
			0.1, 0.8, 0.3, 0.1,
			0.1, -0.5, -0.9, 0.2,
			-0.2, 0.3, -0.4, -0.5,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewDot(x1, x2)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{1.44}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{0.5}))

	if !floats.EqualApprox(x1.grad.Data(), []float64{
		0.05, 0.4, 0.15, 0.05,
		0.05, -0.25, -0.45, 0.1,
		-0.1, 0.15, -0.2, -0.25,
	}, 1.0e-6) {
		t.Error("The x1-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x2.grad.Data(), []float64{
		0.05, 0.1, 0.15, 0.0,
		0.2, 0.25, -0.3, 0.35,
		-0.25, 0.4, -0.4, -0.05,
	}, 1.0e-6) {
		t.Error("The x2-gradients don't match the expected values")
	}
}
