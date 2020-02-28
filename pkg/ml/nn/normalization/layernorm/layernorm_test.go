// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layernorm

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

	x1 := g.NewVariable(mat.NewVecDense([]float64{1.0, 2.0, 0.0, 4.0}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{3.0, 2.0, 1.0, 6.0}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{6.0, 2.0, 5.0, 1.0}), true)

	y := model.NewProc(g).Forward(x1, x2, x3)

	if !floats.EqualApprox(y[0].Value().Data(), []float64{0.1464537236, 0.2661938298, -0.154964787, 1.3170221268}, 1.0e-06) {
		t.Error("The output at position 0 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[1].Value().Data(), []float64{0.4, 0.4069044968, -0.1207134903, 1.3828539612}, 1.0e-06) {
		t.Error("The output at position 1 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[2].Value().Data(), []float64{1.0063390626, 0.445521375, 0.4182820625, -0.8701425001}, 1.0e-06) {
		t.Error("The output at position 2 doesn't match the expected values")
	}

	// == Backward
	// TODO: check gradients
}

func newTestModel() *Model {
	model := New(4)
	model.W.Value().SetData([]float64{0.5, -0.2, 0.3, 0.8})
	model.B.Value().SetData([]float64{0.4, 0.3, 0.2, 0.1})
	return model
}
