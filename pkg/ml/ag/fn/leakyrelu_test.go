// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestLeakyReLUForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	alpha := &variable{
		value:        mat.NewScalar(2.0),
		grad:         nil,
		requiresGrad: false,
	}
	f := NewLeakyReLU(x, alpha)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.1, -0.4, 0.3, 0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-1.0, 1.0, 0.8, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}
