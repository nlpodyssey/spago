// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestDropout_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewDropout(x, 0.25, rand.NewLockedRand(1))
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{
		0.666666, 0.799999, -1.066666, -0.799999, 0.0, -0.5333333, 0.133333, 0.0, 0.399999, -0.666666,
	}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{0.5, 0.4, 0.2, -0.6, 0.3, 0.4, -0.8, -0.3, 0.0, 0.1}))

	if !floats.EqualApprox(x.grad.Data(), []float64{
		0.666666666, 0.533333333, 0.2666666, -0.8, 0.0, 0.5333333, -1.0666666, 0.0, 0.0, 0.133333,
	}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestZeroDropout_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewDropout(x, 0.0, rand.NewLockedRand(1))
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{
		0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5,
	}, 1.0e-6) {
		t.Error("The output don't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{0.5, 0.4, 0.2, -0.6, 0.3, 0.4, -0.8, -0.3, 0.0, 0.1}))

	if !floats.EqualApprox(x.grad.Data(), []float64{
		0.5, 0.4, 0.2, -0.6, 0.3, 0.4, -0.8, -0.3, 0.0, 0.1,
	}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestTotalDropout_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewDropout(x, 1.0, rand.NewLockedRand(1))
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	}, 1.0e-6) {
		t.Error("The output don't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{0.5, 0.4, 0.2, -0.6, 0.3, 0.4, -0.8, -0.3, 0.0, 0.1}))

	if !floats.EqualApprox(x.grad.Data(), []float64{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
	}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}
