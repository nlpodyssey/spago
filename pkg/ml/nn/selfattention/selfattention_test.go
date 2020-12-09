// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"gonum.org/v1/gonum/floats"
	"math"
	"testing"
)

func TestModel_SelfAttention(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training})

	x1 := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{0.8, -0.3, 0.5, 0.3}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{-0.2, 0.7, 0.2, 0.4}), true)

	output := proc.Forward(x1, x2, x3)

	if !floats.EqualApprox(output[0].Value().Data(), []float64{0.789110, -0.755551, -0.431247}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}
	if !floats.EqualApprox(output[1].Value().Data(), []float64{0.780654, -0.6212001, -0.380214}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}
	if !floats.EqualApprox(output[2].Value().Data(), []float64{0.7586521, -0.569575, -0.390976}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	output[0].PropagateGrad(mat.NewVecDense([]float64{-0.04, 0.36, 0.32}))
	output[1].PropagateGrad(mat.NewVecDense([]float64{-0.08, -0.2, -0.1}))
	output[2].PropagateGrad(mat.NewVecDense([]float64{0.1, 0.3, 0.8}))

	g.BackwardAll()

	if !floats.EqualApprox(x1.Grad().Data(), []float64{0.01654154, 0.48942297, -0.1587743, -0.2387454}, 1.0e-05) {
		t.Error("The input gradients x1 don't match the expected values")
	}
	if !floats.EqualApprox(x2.Grad().Data(), []float64{0.04408108, 0.18716132, -0.15425818, -0.040870}, 1.0e-05) {
		t.Error("The input gradients x2 don't match the expected values")
	}
	if !floats.EqualApprox(x3.Grad().Data(), []float64{0.05800791, 0.205784865, -0.2431444, -0.1281430}, 1.0e-05) {
		t.Error("The input gradients x3 don't match the expected values")
	}

	if !floats.EqualApprox(model.Value.W.Grad().(*mat.Dense).Data(), []float64{
		0.01790323, 0.01853984, 0.01959262, -0.020789988,
		-0.1529254, -0.2133677, -0.17563661, 0.336455424,
		-0.2887047, -0.3777314, -0.31337569, 0.705232107,
	}, 1.0e-05) {
		t.Error("Value W doesn't match the expected values")
	}
	if !floats.EqualApprox(model.Value.B.Grad().Data(), []float64{
		-0.02, 0.46, 1.02,
	}, 1.0e-05) {
		t.Error("Value B doesn't match the expected values")
	}
	if !floats.EqualApprox(model.Key.W.Grad().(*mat.Dense).Data(), []float64{
		0.07438275, 0.15194683, 0.11696175, -0.0629919,
		0.03235329, 0.05018469, 0.04422187, -0.0234946,
		-0.0599427, -0.1594204, -0.1097165, 0.0598379,
	}, 1.0e-05) {
		t.Error("Key W doesn't match the expected values")
	}
	if !floats.EqualApprox(model.Key.B.Grad().Data(), []float64{
		0, 0, 0,
	}, 1.0e-05) {
		t.Error("Key B doesn't match the expected values")
	}
	if !floats.EqualApprox(model.Query.W.Grad().(*mat.Dense).Data(), []float64{
		0.00538138, -0.0264289, -0.0085512, -0.0088408,
		-0.0175901, -0.0032803, -0.0132455, 0.0143783,
		-0.1022365, -0.0221910, -0.0784209, 0.08306303,
	}, 1.0e-05) {
		t.Error("Query W doesn't match the expected values")
	}
	if !floats.EqualApprox(model.Query.B.Grad().Data(), []float64{
		-0.0267918, 0.0149118, 0.08413719,
	}, 1.0e-05) {
		t.Error("Query B doesn't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(Config{
		InputSize:   4,
		QuerySize:   3,
		KeySize:     3,
		ValueSize:   3,
		ScaleFactor: 1.0 / math.Sqrt(3.0),
	})
	model.Value.W.Value().SetData([]float64{
		0.5, 0.6, -0.8, 0.7,
		-0.4, 0.1, 0.7, -0.7,
		0.3, 0.8, -0.9, 0.0,
	})
	model.Value.B.Value().SetData([]float64{0.4, 0.0, -0.3})
	model.Key.W.Value().SetData([]float64{
		0.7, -0.2, -0.1, 0.2,
		-0.1, -0.1, 0.3, -0.2,
		0.6, 0.1, 0.9, 0.3,
	})
	model.Key.B.Value().SetData([]float64{0.8, -0.2, -0.5})
	model.Query.W.Value().SetData([]float64{
		-0.8, -0.6, 0.2, 0.5,
		0.7, -0.6, -0.3, 0.6,
		-0.3, 0.3, 0.4, -0.8,
	})
	model.Query.B.Value().SetData([]float64{0.3, 0.5, -0.7})
	return model
}
