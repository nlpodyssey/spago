// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rmsnorm

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
	x1 := g.NewVariable(mat.NewVecDense([]float64{1.0, 2.0, 0.0, 4.0}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{3.0, 2.0, 1.0, 6.0}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{6.0, 2.0, 5.0, 1.0}), true)

	y := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x1, x2, x3)

	if !floats.EqualApprox(y[0].Value().Data(), []float64{0.6182178902, 0.1254256878, 0.2, 1.4965944974}, 1.0e-06) {
		t.Error("The output at position 0 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[1].Value().Data(), []float64{0.8242640687, 0.186862915, 0.2848528137, 1.4576450198}, 1.0e-06) {
		t.Error("The output at position 1 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[2].Value().Data(), []float64{1.1385489459, 0.2015268072, 0.5692744729, 0.2969463856}, 1.0e-06) {
		t.Error("The output at position 2 doesn't match the expected values")
	}

	// == Backward
	y[0].PropagateGrad(mat.NewVecDense([]float64{-1.0, -0.2, 0.4, 0.6}))
	y[1].PropagateGrad(mat.NewVecDense([]float64{-0.3, 0.1, 0.7, 0.9}))
	y[2].PropagateGrad(mat.NewVecDense([]float64{0.3, -0.4, 0.7, -0.8}))
	g.BackwardAll()

	if !floats.EqualApprox(x1.Grad().Data(), []float64{-0.2493918746, -0.0448905374, 0.0523722937, 0.0847932373}, 1.0e-06) {
		t.Error("The x1-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{-0.1109874804, -0.0513642366, 0.0365432785, 0.066524606}, 1.0e-06) {
		t.Error("The x2-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x3.Grad().Data(), []float64{0.0040284488, 0.0087283057, 0.0242825941, -0.1630402749}, 1.0e-06) {
		t.Error("The x3-gradients don't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(4)
	model.W.Value().SetData([]float64{0.5, -0.2, 0.3, 0.8})
	model.B.Value().SetData([]float64{0.4, 0.3, 0.2, 0.1})
	return model
}
