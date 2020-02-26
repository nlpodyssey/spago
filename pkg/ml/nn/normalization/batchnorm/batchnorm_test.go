// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package batchnorm

import (
	"gonum.org/v1/gonum/floats"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/ag"
	"testing"
)

func TestModel_Forward(t *testing.T) {

	model := newTestModel()
	g := ag.NewGraph()

	// == Forward

	x1 := g.NewVariable(mat.NewVecDense([]float64{0.4, 0.8, -0.7, -0.5}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{-0.4, -0.6, -0.2, -0.9}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{0.4, 0.4, 0.2, 0.8}), true)

	y := rectify(g, model.NewProc(g).Forward(x1, x2, x3)) // TODO: rewrite tests without activation function

	if !floats.EqualApprox(y[0].Value().Data(), []float64{1.1828427, 0.2, 0.0, 0.0}, 1.0e-06) {
		t.Error("The output at position 0 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[1].Value().Data(), []float64{0.334314, 0.2, 0.0, 0.0}, 1.0e-06) {
		t.Error("The output at position 1 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[2].Value().Data(), []float64{1.1828427, 0.2, 0.0, 1.302356}, 1.0e-06) {
		t.Error("The output at position 2 doesn't match the expected values")
	}

	// == Backward

	y[0].PropagateGrad(mat.NewVecDense([]float64{-1.0, -0.2, 0.4, 0.6}))
	y[1].PropagateGrad(mat.NewVecDense([]float64{-0.3, 0.1, 0.7, 0.9}))
	y[2].PropagateGrad(mat.NewVecDense([]float64{0.3, -0.4, 0.7, -0.8}))
	g.BackwardAll()

	if !floats.EqualApprox(x1.Grad().Data(), []float64{-0.6894291116772131, 0.0, 0.0, 0.1265151774227913}, 1.0e-06) {
		t.Error("The x1-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{-1.767774815419898e-11, 0.0, 0.0, -0.09674690039596812}, 1.0e-06) {
		t.Error("The x2-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x3.Grad().Data(), []float64{0.6894291116595355, 0.0, 0.0, -0.029768277056219317}, 1.0e-06) {
		t.Error("The x3-gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.B.Grad().Data(), []float64{-1.0, -0.5, 0.0, -0.8}, 1.0e-06) {
		t.Error("The biases B doesn't match the expected values")
	}

	if !floats.EqualApprox(model.W.Grad().Data(), []float64{-0.070710, -0.475556, 0.0, -1.102356}, 1.0e-06) {
		t.Error("The weights W doesn't match the expected values")
	}
}

func rectify(g *ag.Graph, xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = g.ReLU(x)
	}
	return ys
}

func newTestModel() *Model {
	model := New(4)
	model.W.Value().SetData([]float64{0.4, 0.0, -0.3, 0.8})
	model.B.Value().SetData([]float64{0.9, 0.2, -0.9, 0.2})
	return model
}
