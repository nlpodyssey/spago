// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/saientist/spago/pkg/mat"
	"github.com/saientist/spago/pkg/ml/ag"
	"gonum.org/v1/gonum/floats"
	"math"
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

func TestScaledDotProductAttention(t *testing.T) {
	g := ag.NewGraph()
	qs := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{1.1, 0.0, 2.3}), true),
		g.NewVariable(mat.NewVecDense([]float64{2.2, -0.5, 0.3}), true),
		g.NewVariable(mat.NewVecDense([]float64{3.2, 0.5, 0.4}), true),
	}
	ks := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{0.0, 1.2, 1.3}), true),
		g.NewVariable(mat.NewVecDense([]float64{4.5, 4.3, 0.2}), true),
		g.NewVariable(mat.NewVecDense([]float64{2.7, 3.6, 2.1}), true),
	}
	vs := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{1.2, 2.3, 3.4}), true),
		g.NewVariable(mat.NewVecDense([]float64{2.2, 8.5, 0.0}), true),
		g.NewVariable(mat.NewVecDense([]float64{2.3, 6.5, 3.5}), true),
	}

	context, _ := ScaledDotProductAttention(g, qs, ks, vs, math.Sqrt(3))

	if len(context) != 3 {
		t.Error("The attention doesn't have the expected length")
	}
	if !floats.EqualApprox(context[0].Value().Data(), []float64{2.22875441063165, 6.68411289826994, 2.82497984315079}, 1.0e-6) {
		t.Error("Attention[0] doesn't match the expected values")
	}
	if !floats.EqualApprox(context[1].Value().Data(), []float64{2.20637295180029, 8.15650999969648, 0.539678848469417}, 1.0e-6) {
		t.Error("Attention[1] doesn't match the expected values")
	}
	if !floats.EqualApprox(context[2].Value().Data(), []float64{2.20423303670527, 8.41210390591632, 0.152898186332002}, 1.0e-6) {
		t.Error("Attention[2] doesn't match the expected values")
	}
}
