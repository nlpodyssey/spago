// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/saientist/spago/pkg/mat"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestConcat_Forward(t *testing.T) {
	x1 := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3}),
		grad:         nil,
		requiresGrad: true,
	}
	x2 := &variable{
		value:        mat.NewVecDense([]float64{0.4, 0.5, 0.6, 0.7}),
		grad:         nil,
		requiresGrad: true,
	}
	x3 := &variable{
		value:        mat.NewVecDense([]float64{0.8, 0.9}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewConcat([]Operand{x1, x2, x3})
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}))

	if !floats.EqualApprox(x1.grad.Data(), []float64{1.0, 2.0, 3.0}, 1.0e-6) {
		t.Error("The x1-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x2.grad.Data(), []float64{4.0, 5.0, 6.0, 7.0}, 1.0e-6) {
		t.Error("The x2-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x3.grad.Data(), []float64{8.0, 9.0}, 1.0e-6) {
		t.Error("The x3-gradients don't match the expected values")
	}
}
