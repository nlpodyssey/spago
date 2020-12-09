// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModelReLU_Forward(t *testing.T) {
	g := ag.NewGraph()

	// == Forward
	x := g.NewVariable(mat.NewVecDense([]float64{0.1, -0.2, 0.3, 0.0}), true)
	y := New(ag.OpReLU).NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x)[0]

	if !floats.EqualApprox(y.Value().Data(), []float64{0.1, 0.0, 0.3, 0.0}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward
	g.Backward(y, ag.OutputGrad(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0})))

	if !floats.EqualApprox(x.Grad().Data(), []float64{-1.0, 0.0, 0.8, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestModelSwish_Forward(t *testing.T) {
	g := ag.NewGraph()

	beta := nn.NewParam(mat.NewScalar(2.0))
	model := New(ag.OpSwish, beta)

	// == Forward
	x := g.NewVariable(mat.NewVecDense([]float64{0.1, -0.2, 0.3, 0.0}), true)
	y := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x)[0]

	if !floats.EqualApprox(y.Value().Data(), []float64{0.0549833997, -0.080262468, 0.1936968919, 0.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward
	g.Backward(y, ag.OutputGrad(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0})))

	if !floats.EqualApprox(x.Grad().Data(), []float64{-0.5993373119, 0.1526040208, 0.6263414804, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}

	if !floats.EqualApprox(beta.Grad().Data(), []float64{0.0188025145}, 1.0e-6) {
		t.Error("The beta-gradients don't match the expected values")
	}
}
