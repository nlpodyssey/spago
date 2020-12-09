// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layernorm

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()

	// == Forward
	x := g.NewVariable(mat.NewVecDense([]float64{0.4, 0.8, -0.7, -0.5}), true)
	y := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x)[0]

	if !floats.EqualApprox(y.Value().Data(), []float64{1.157863, 0.2, -0.561554, -0.444658}, 1.0e-06) {
		t.Error("The output at position 0 doesn't match the expected values")
	}

	// == Backward
	y.PropagateGrad(mat.NewVecDense([]float64{-1.0, -0.2, 0.4, 0.6}))
	g.BackwardAll()

	if !floats.EqualApprox(x.Grad().Data(), []float64{-0.496261, 0.280677, -0.408772, 0.624355}, 1.0e-06) {
		t.Error("The x1-gradients don't match the expected values")
	}
	if !floats.EqualApprox(model.W.Grad().Data(), []float64{-0.644658, -0.257863, -0.45126, -0.483493}, 1.0e-06) {
		t.Error("The W-gradients don't match the expected values")
	}
	if !floats.EqualApprox(model.B.Grad().Data(), []float64{-1.0, -0.2, 0.4, 0.6}, 1.0e-06) {
		t.Error("The B-gradients don't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(4)
	model.W.Value().SetData([]float64{0.4, 0.0, -0.3, 0.8})
	model.B.Value().SetData([]float64{0.9, 0.2, -0.9, 0.2})
	return model
}
