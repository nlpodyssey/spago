// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestSoftmax_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{-0.41, -1.08, 0, 0.87, -0.19, -0.75}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSoftmax(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.1166451, 0.0596882, 0.1757629, 0.4195304, 0.1453487, 0.083024}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{0.0, 0.0, -5.689482, 0.0, 0.0, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{0.1166451, 0.0596882, -0.8242370, 0.4195304, 0.1453487, 0.083024}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}
