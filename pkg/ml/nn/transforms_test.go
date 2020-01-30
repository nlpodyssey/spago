// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"gonum.org/v1/gonum/floats"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/ag"
	"testing"
)

func TestConv2D(t *testing.T) {
	var g = ag.NewGraph()

	x := g.NewVariable(mat.NewDense(4, 4, []float64{
		0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.3,
		0.5, -0.6, -0.4, 0.6,
		-0.3, 0.9, 0.5, 0.5,
	}), true)

	w := g.NewVariable(mat.NewDense(2, 2, []float64{
		0.5, -0.4,
		0.3, 0.3,
	}), true)

	out := Conv2D(g, w, x, 1, 1)

	if !floats.EqualApprox(out.Value().Data(), []float64{
		0.09, -0.3, -0.22,
		0.29, -0.37, 0.08,
		0.67, 0.28, -0.14,
	}, 0.005) {
		t.Error("out value doesn't match the expected values")
	}

	g.Backward(out, mat.NewDense(3, 3, []float64{
		1.0, -0.5, -1.0,
		0.5, 0.3, 0.5,
		0.2, 0.5, -0.5,
	}))

	if !floats.EqualApprox(w.Grad().Data(), []float64{
		-0.34, -1.93,
		0.76, 0.16,
	}, 0.005) {
		t.Error("w gradients don't match the expected values")
	}

	if !floats.EqualApprox(x.Grad().Data(), []float64{
		0.5, -0.65, -0.3, 0.4,
		0.55, 0.1, -0.32, -0.5,
		0.25, 0.41, -0.21, 0.35,
		0.06, 0.21, 0.0, -0.15,
	}, 0.005) {
		t.Error("x gradients don't match the expected values")
	}
}

func TestConv2DStride2(t *testing.T) {

	var g = ag.NewGraph()

	x := g.NewVariable(mat.NewDense(4, 4, []float64{
		0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.3,
		0.5, -0.6, -0.4, 0.6,
		-0.3, 0.9, 0.5, 0.5,
	}), true)

	w := g.NewVariable(mat.NewDense(2, 2, []float64{
		0.5, -0.4,
		0.3, 0.3,
	}), true)

	out := Conv2D(g, w, x, 2, 2)

	if !floats.EqualApprox(out.Value().Data(), []float64{
		0.09, -0.22,
		0.67, -0.14,
	}, 0.005) {
		t.Error("out value doesn't match the expected values")
	}

	g.Backward(out, mat.NewDense(2, 2, []float64{
		1.0, -0.5,
		0.5, 0.3,
	}))

	if !floats.EqualApprox(w.Grad().Data(), []float64{
		0.08, -0.42,
		0.5, 0.45,
	}, 0.005) {
		t.Error("w gradients don't match the expected values")
	}

	if !floats.EqualApprox(x.Grad().Data(), []float64{
		0.5, -0.4, -0.25, 0.2,
		0.3, 0.3, -0.15, -0.15,
		0.25, -0.2, 0.15, -0.12,
		0.15, 0.15, 0.09, 0.09,
	}, 0.005) {
		t.Error("x gradients don't match the expected values")
	}
}
