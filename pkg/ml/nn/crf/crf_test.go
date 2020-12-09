// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crf

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"gonum.org/v1/gonum/floats"
	"reflect"
	"testing"
)

func TestModel_Predict(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()

	w1 := g.NewVariable(mat.NewVecDense([]float64{1.7, 0.2, -0.3, 0.5}), true)
	w2 := g.NewVariable(mat.NewVecDense([]float64{2.0, -3.5, 0.1, 2.0}), true)
	w3 := g.NewVariable(mat.NewVecDense([]float64{-2.5, 3.2, -0.2, -0.3}), true)
	w4 := g.NewVariable(mat.NewVecDense([]float64{3.3, -0.9, 2.7, -2.7}), true)
	w5 := g.NewVariable(mat.NewVecDense([]float64{0.5, 0.2, 0.4, 1.4}), true)

	y := model.Predict([]ag.Node{w1, w2, w3, w4, w5})

	gold := []int{3, 3, 1, 0, 3}

	if !reflect.DeepEqual(y, gold) {
		t.Error("Predictions don't match the expected values")
	}
}

func TestModel_GoldScore(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training})

	w1 := g.NewVariable(mat.NewVecDense([]float64{1.7, 0.2, -0.3, 0.5}), true)
	w2 := g.NewVariable(mat.NewVecDense([]float64{2.0, -3.5, 0.1, 2.0}), true)
	w3 := g.NewVariable(mat.NewVecDense([]float64{-2.5, 3.2, -0.2, -0.3}), true)
	w4 := g.NewVariable(mat.NewVecDense([]float64{3.3, -0.9, 2.7, -2.7}), true)
	w5 := g.NewVariable(mat.NewVecDense([]float64{0.5, 0.2, 0.4, 1.4}), true)

	gold := []int{0, 0, 1, 0, 3}
	y := proc.(*Processor).goldScore([]ag.Node{w1, w2, w3, w4, w5}, gold)

	if !floats.EqualApprox(y.Value().Data(), []float64{14.27}, 0.000001) {
		t.Error("Predictions don't match the expected values")
	}
}

func TestModel_TotalScore(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training})

	w1 := g.NewVariable(mat.NewVecDense([]float64{1.7, 0.2, -0.3, 0.5}), true)
	w2 := g.NewVariable(mat.NewVecDense([]float64{2.0, -3.5, 0.1, 2.0}), true)
	w3 := g.NewVariable(mat.NewVecDense([]float64{-2.5, 3.2, -0.2, -0.3}), true)
	w4 := g.NewVariable(mat.NewVecDense([]float64{3.3, -0.9, 2.7, -2.7}), true)
	w5 := g.NewVariable(mat.NewVecDense([]float64{0.5, 0.2, 0.4, 1.4}), true)

	y := proc.(*Processor).totalScore([]ag.Node{w1, w2, w3, w4, w5})

	if !floats.EqualApprox(y.Value().Data(), []float64{16.64258452}, 0.000001) {
		t.Error("Total score doesn't match the expected values")
	}
}

func TestModel_Loss(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training})

	w1 := g.NewVariable(mat.NewVecDense([]float64{1.7, 0.2, -0.3, 0.5}), true)
	w2 := g.NewVariable(mat.NewVecDense([]float64{2.0, -3.5, 0.1, 2.0}), true)
	w3 := g.NewVariable(mat.NewVecDense([]float64{-2.5, 3.2, -0.2, -0.3}), true)
	w4 := g.NewVariable(mat.NewVecDense([]float64{3.3, -0.9, 2.7, -2.7}), true)
	w5 := g.NewVariable(mat.NewVecDense([]float64{0.5, 0.2, 0.4, 1.4}), true)

	gold := []int{0, 0, 1, 0, 3}
	loss := proc.(*Processor).NegativeLogLoss([]ag.Node{w1, w2, w3, w4, w5}, gold)

	g.Backward(loss)
	if !floats.EqualApprox(loss.Value().Data(), []float64{2.37258452}, 0.000001) {
		t.Error("Total score doesn't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(4)
	model.TransitionScores.Value().SetData([]float64{
		0.0, 0.6, 0.8, 1.2, 1.6,
		0.2, 0.5, 0.02, 0.03, 0.45,
		0.3, 0.2, 0.6, 0.01, 0.19,
		0.4, 0.02, 0.02, 0.7, 0.26,
		0.9, 0.1, 0.02, 0.08, 0.8,
	})
	return model
}
