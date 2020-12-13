// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
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

	g.Backward(out, ag.OutputGrad(mat.NewDense(3, 3, []float64{
		1.0, -0.5, -1.0,
		0.5, 0.3, 0.5,
		0.2, 0.5, -0.5,
	})))

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

	g.Backward(out, ag.OutputGrad(mat.NewDense(2, 2, []float64{
		1.0, -0.5,
		0.5, 0.3,
	})))

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

	context, _ := ScaledDotProductAttention(g, qs, ks, vs, 1.0/math.Sqrt(3), false)

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

func TestScaledDotProductAttention2(t *testing.T) {
	g := ag.NewGraph()
	qs := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{0.22, 0.3}), true),
		g.NewVariable(mat.NewVecDense([]float64{-0.17, 0.24}), true),
		g.NewVariable(mat.NewVecDense([]float64{-0.15, 0.23}), true),
	}
	ks := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{1.66, 0.12}), true),
		g.NewVariable(mat.NewVecDense([]float64{0.88, -0.02}), true),
		g.NewVariable(mat.NewVecDense([]float64{-0.3, -0.46}), true),
	}
	vs := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{0.83, 0.7, -0.25, -0.58}), true),
		g.NewVariable(mat.NewVecDense([]float64{0.0, 0.2, 0.57, -2.08}), true),
		g.NewVariable(mat.NewVecDense([]float64{-0.07, 0.0, 0.29, 0.5}), true),
	}

	// == Forward
	context, probs := ScaledDotProductAttention(g, qs, ks, vs, 1.0/math.Sqrt(2), false)

	if len(context) != 3 {
		t.Error("The context doesn't have the expected length")
	}
	if len(probs) != 3 {
		t.Error("The probs doesn't have the expected length")
	}
	if !floats.EqualApprox(context[0].Value().Data(), []float64{0.312291, 0.347165, 0.170855, -0.813202}, 1.0e-6) {
		t.Error("Context[0] doesn't match the expected values")
	}
	if !floats.EqualApprox(context[1].Value().Data(), []float64{0.232861, 0.284047, 0.21555, -0.694914}, 1.0e-6) {
		t.Error("Context[1] doesn't match the expected values")
	}
	if !floats.EqualApprox(context[2].Value().Data(), []float64{0.236194, 0.28672, 0.21373, -0.700304}, 1.0e-6) {
		t.Error("Context[2] doesn't match the expected values")
	}
	if !floats.EqualApprox(probs[0].Data(), []float64{0.398142, 0.342329, 0.259529}, 1.0e-6) {
		t.Error("Probs[0] doesn't match the expected values")
	}
	if !floats.EqualApprox(probs[1].Data(), []float64{0.310603, 0.333125, 0.356272}, 1.0e-6) {
		t.Error("Probs[1] doesn't match the expected values")
	}
	if !floats.EqualApprox(probs[2].Data(), []float64{0.314262, 0.333682, 0.352055}, 1.0e-6) {
		t.Error("Probs[2] doesn't match the expected values")
	}

	// == Backward
	context[0].PropagateGrad(mat.NewVecDense([]float64{0.7, -0.3, -0.7, -0.5}))
	context[1].PropagateGrad(mat.NewVecDense([]float64{-0.8, -0.5, -0.5, 0.1}))
	context[2].PropagateGrad(mat.NewVecDense([]float64{-0.6, -0.5, 0.2, -0.9}))
	g.BackwardAll()

	if !floats.EqualApprox(qs[0].Grad().Data(), []float64{0.291064, 0.090078}, 1.0e-6) {
		t.Error("qs[0] doesn't match the expected values")
	}
	if !floats.EqualApprox(qs[1].Grad().Data(), []float64{-0.214319, -0.065291}, 1.0e-6) {
		t.Error("qs[1] doesn't match the expected values")
	}
	if !floats.EqualApprox(qs[2].Grad().Data(), []float64{0.084357, 0.057063}, 1.0e-6) {
		t.Error("qs[2] doesn't match the expected values")
	}

	if !floats.EqualApprox(ks[0].Grad().Data(), []float64{0.06886, -0.025612}, 1.0e-6) {
		t.Error("ks[0] doesn't match the expected values")
	}
	if !floats.EqualApprox(ks[1].Grad().Data(), []float64{-0.039958, 0.089393}, 1.0e-6) {
		t.Error("ks[1] doesn't match the expected values")
	}
	if !floats.EqualApprox(ks[2].Grad().Data(), []float64{-0.028902, -0.063781}, 1.0e-6) {
		t.Error("ks[2] doesn't match the expected values")
	}

	if !floats.EqualApprox(vs[0].Grad().Data(), []float64{-0.15834, -0.431875, -0.371149, -0.450847}, 1.0e-6) {
		t.Error("vs[0] doesn't match the expected values")
	}
	if !floats.EqualApprox(vs[1].Grad().Data(), []float64{-0.22708, -0.436103, -0.339456, -0.438166}, 1.0e-6) {
		t.Error("vs[1] doesn't match the expected values")
	}
	if !floats.EqualApprox(vs[2].Grad().Data(), []float64{-0.31458, -0.432022, -0.289395, -0.410987}, 1.0e-6) {
		t.Error("vs[2] doesn't match the expected values")
	}
}

func TestLinearAttention(t *testing.T) {
	g := ag.NewGraph()
	qs := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{1.8, 1.35, -1.89}), true),
		g.NewVariable(mat.NewVecDense([]float64{0.08, 1.27, -1.06}), true),
		g.NewVariable(mat.NewVecDense([]float64{0.28, 0.12, -0.67}), true),
	}
	ks := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{0.71, -0.5, -1.58}), true),
		g.NewVariable(mat.NewVecDense([]float64{1.43, -0.16, 0.49}), true),
		g.NewVariable(mat.NewVecDense([]float64{0.58, -0.27, -0.25}), true),
	}
	vs := []ag.Node{
		g.NewVariable(mat.NewVecDense([]float64{0.88, -1.09, -0.45}), true),
		g.NewVariable(mat.NewVecDense([]float64{0.43, -0.21, -0.75}), true),
		g.NewVariable(mat.NewVecDense([]float64{0.84, 0.01, 0.01}), true),
	}

	defaultMappingFunction := func(g *ag.Graph, x ag.Node) ag.Node {
		return g.PositiveELU(x)
	}
	output := LinearAttention(g, qs, ks, vs, defaultMappingFunction, 1e-12)

	if len(output) != 3 {
		t.Error("The attention doesn't have the expected length")
	}
	if !floats.EqualApprox(output[0].Value().Data(), []float64{0.68021652, -0.39977211, -0.44051976}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}
	if !floats.EqualApprox(output[1].Value().Data(), []float64{0.678651, -0.38249578, -0.43479299}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}
	if !floats.EqualApprox(output[2].Value().Data(), []float64{0.6720585, -0.38117003, -0.44469679}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}
}
