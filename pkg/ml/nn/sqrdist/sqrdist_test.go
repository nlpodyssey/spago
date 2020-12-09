// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sqrdist

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

	x := g.NewVariable(mat.NewVecDense([]float64{0.3, 0.5, -0.4}), true)
	y := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x)[0]

	if !floats.EqualApprox(y.Value().Data(), []float64{0.5928}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	g.Backward(y, ag.OutputGrad(mat.NewScalar(-0.8)))

	if !floats.EqualApprox(x.Grad().Data(), []float64{-0.9568, -0.848, 0.5936}, 1.0e-05) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.B.Grad().(*mat.Dense).Data(), []float64{
		-0.2976, -0.496, 0.3968,
		0.0144, 0.024, -0.0192,
		-0.1488, -0.248, 0.1984,
		-0.1584, -0.264, 0.2112,
		0.024, 0.04, -0.032,
	}, 1.0e-06) {
		t.Error("WIn doesn't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(3, 5)
	model.B.Value().SetData([]float64{
		0.4, 0.6, -0.5,
		-0.5, 0.4, 0.2,
		0.5, 0.4, 0.1,
		0.5, 0.2, -0.2,
		-0.3, 0.4, 0.4,
	})
	return model
}
