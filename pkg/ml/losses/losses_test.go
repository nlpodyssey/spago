// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package losses

import (
	"gonum.org/v1/gonum/floats"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/ag"
	"testing"
)

func TestMSELoss(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]float64{0.0, 0.1, 0.2, 0.3}), true)
	y := g.NewVariable(mat.NewVecDense([]float64{0.3, 0.2, 0.1, 0.0}), false)
	loss := MSE(g, x, y, false)

	if !equalApprox(loss.Value().Scalar(), 0.1) {
		t.Error("The loss doesn't match the expected value")
	}

	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{-0.3, -0.1, 0.1, 0.3}, 1.0e-6) {
		t.Error("The gradients don't match the expected values")
	}
}

func TestNLLLoss(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]float64{-500, 0, 0.693147, 1.94591}), true)
	y := g.NewVariable(mat.NewVecDense([]float64{0.0, 0.0, 1.0, 0.0}), false)
	loss := NLL(g, g.Softmax(x), y)

	if !equalApprox(loss.Value().Scalar(), 1.609438) {
		t.Error("The loss doesn't match the expected value")
	}

	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.0, 0.1, -0.8, 0.7}, 1.0e-6) {
		t.Error("The gradients don't match the expected values")
	}
}

func TestCrossEntropyLoss(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]float64{-500, 0, 0.693147, 1.94591}), true)
	loss := CrossEntropy(g, x, 2)

	if !equalApprox(loss.Value().Scalar(), 1.609438) {
		t.Error("The loss doesn't match the expected value")
	}

	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.0, 0.1, -0.8, 0.7}, 1.0e-6) {
		t.Error("The gradients don't match the expected values")
	}
}

func TestZeroOneQuantization(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]float64{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), true)
	loss := ZeroOneQuantization(g, x)

	if !equalApprox(loss.Value().Scalar(), 2.209) {
		t.Error("The loss doesn't match the expected value")
	}

	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.144, 0.192, 0.0, 0.096, -7.488, 0.168}, 1.0e-6) {
		t.Error("The gradients don't match the expected values")
	}
}

func TestNorm2Quantization(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]float64{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), true)
	loss := Norm2Quantization(g, x)

	if !equalApprox(loss.Value().Scalar(), 0.8836) {
		t.Error("The loss doesn't match the expected value")
	}

	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.376, 0.752, 3.76, 1.504, -3.008, 1.128}, 1.0e-6) {
		t.Error("The gradients don't match the expected values")
	}
}

func TestOneHotQuantization(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]float64{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), true)
	loss := OneHotQuantization(g, x, 0.1)

	if !equalApprox(loss.Value().Scalar(), 0.30926) {
		t.Error("The loss doesn't match the expected value")
	}

	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.052, 0.0944, 0.376, 0.16, -1.0496, 0.1296}, 1.0e-6) {
		t.Error("The gradients don't match the expected values")
	}
}

func TestMSESeqLoss(t *testing.T) {
	g := ag.NewGraph()
	x1 := g.NewVariable(mat.NewVecDense([]float64{0.0, 0.1, 0.2, 0.3}), true)
	y1 := g.NewVariable(mat.NewVecDense([]float64{0.3, 0.2, 0.1, 0.0}), false)
	x2 := g.NewVariable(mat.NewVecDense([]float64{0.0, 0.1, 0.2, 0.3}), true)
	y2 := g.NewVariable(mat.NewVecDense([]float64{0.3, 0.2, 0.1, 0.0}), false)
	loss := MSESeq(g, []ag.Node{x1, x2}, []ag.Node{y1, y2}, true)

	if !equalApprox(loss.Value().Scalar(), 0.1) {
		t.Error("The loss doesn't match the expected value")
	}

	g.Backward(loss)

	if !floats.EqualApprox(x1.Grad().Data(), []float64{-0.15, -0.05, 0.05, 0.15}, 1.0e-6) {
		t.Error("The x1-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{-0.15, -0.05, 0.05, 0.15}, 1.0e-6) {
		t.Error("The x2-gradients don't match the expected values")
	}
}

func equalApprox(a, b float64) bool {
	return floats.EqualWithinAbsOrRel(a, b, 1.0e-06, 1.0e-06)
}
