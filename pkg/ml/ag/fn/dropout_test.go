// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/saientist/spago/pkg/mat"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestDropout_Forward(t *testing.T) {

	x := &variable{
		value:        mat.NewVecDense([]float64{0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewDropout(x, 0.25, rand.NewSource(1))
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{
		0.666666, 0.799999, -1.066666, -0.799999, 0.0, -0.5333333, 0.133333, 0.0, 0.399999, -0.666666, // TODO: check values
	}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	// TODO: check backward
}

func TestZeroDropout_Forward(t *testing.T) {

	x := &variable{
		value:        mat.NewVecDense([]float64{0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewDropout(x, 0.0, rand.NewSource(1))
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{
		0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5, // TODO: check values
	}, 1.0e-6) {
		t.Error("The output don't match the expected values")
	}

	// TODO: check backward
}

func TestTotalDropout_Forward(t *testing.T) {

	x := &variable{
		value:        mat.NewVecDense([]float64{0.5, 0.6, -0.8, -0.6, 0.7, -0.4, 0.1, -0.8, 0.3, -0.5}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewDropout(x, 1.0, rand.NewSource(1))
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // TODO: check values
	}, 1.0e-6) {
		t.Error("The output don't match the expected values")
	}

	// TODO: check backward
}
