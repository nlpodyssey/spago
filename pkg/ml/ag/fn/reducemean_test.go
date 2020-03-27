// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestReduceMean_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewReduceMean(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.15}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{0.5}))

	if !floats.EqualApprox(x.grad.Data(), []float64{0.125, 0.125, 0.125, 0.125}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}
