// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crf

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_Decode(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)

	w1 := g.NewVariable(mat.NewVecDense([]mat.Float{1.7, 0.2, -0.3, 0.5}), true)
	w2 := g.NewVariable(mat.NewVecDense([]mat.Float{2.0, -3.5, 0.1, 2.0}), true)
	w3 := g.NewVariable(mat.NewVecDense([]mat.Float{-2.5, 3.2, -0.2, -0.3}), true)
	w4 := g.NewVariable(mat.NewVecDense([]mat.Float{3.3, -0.9, 2.7, -2.7}), true)
	w5 := g.NewVariable(mat.NewVecDense([]mat.Float{0.5, 0.2, 0.4, 1.4}), true)

	y := proc.Decode([]ag.Node{w1, w2, w3, w4, w5})

	gold := []int{3, 3, 1, 0, 3}

	assert.Equal(t, gold, y)
}

func TestModel_GoldScore(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)

	w1 := g.NewVariable(mat.NewVecDense([]mat.Float{1.7, 0.2, -0.3, 0.5}), true)
	w2 := g.NewVariable(mat.NewVecDense([]mat.Float{2.0, -3.5, 0.1, 2.0}), true)
	w3 := g.NewVariable(mat.NewVecDense([]mat.Float{-2.5, 3.2, -0.2, -0.3}), true)
	w4 := g.NewVariable(mat.NewVecDense([]mat.Float{3.3, -0.9, 2.7, -2.7}), true)
	w5 := g.NewVariable(mat.NewVecDense([]mat.Float{0.5, 0.2, 0.4, 1.4}), true)

	gold := []int{0, 0, 1, 0, 3}
	y := proc.goldScore([]ag.Node{w1, w2, w3, w4, w5}, gold)

	assert.InDeltaSlice(t, []mat.Float{14.27}, y.Value().Data(), 0.00001)
}

func TestModel_TotalScore(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)

	w1 := g.NewVariable(mat.NewVecDense([]mat.Float{1.7, 0.2, -0.3, 0.5}), true)
	w2 := g.NewVariable(mat.NewVecDense([]mat.Float{2.0, -3.5, 0.1, 2.0}), true)
	w3 := g.NewVariable(mat.NewVecDense([]mat.Float{-2.5, 3.2, -0.2, -0.3}), true)
	w4 := g.NewVariable(mat.NewVecDense([]mat.Float{3.3, -0.9, 2.7, -2.7}), true)
	w5 := g.NewVariable(mat.NewVecDense([]mat.Float{0.5, 0.2, 0.4, 1.4}), true)

	y := proc.totalScore([]ag.Node{w1, w2, w3, w4, w5})

	assert.InDeltaSlice(t, []mat.Float{16.64258}, y.Value().Data(), 0.00001)
}

func TestModel_Loss(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)

	w1 := g.NewVariable(mat.NewVecDense([]mat.Float{1.7, 0.2, -0.3, 0.5}), true)
	w2 := g.NewVariable(mat.NewVecDense([]mat.Float{2.0, -3.5, 0.1, 2.0}), true)
	w3 := g.NewVariable(mat.NewVecDense([]mat.Float{-2.5, 3.2, -0.2, -0.3}), true)
	w4 := g.NewVariable(mat.NewVecDense([]mat.Float{3.3, -0.9, 2.7, -2.7}), true)
	w5 := g.NewVariable(mat.NewVecDense([]mat.Float{0.5, 0.2, 0.4, 1.4}), true)

	gold := []int{0, 0, 1, 0, 3}
	loss := proc.NegativeLogLoss([]ag.Node{w1, w2, w3, w4, w5}, gold)

	g.Backward(loss)
	assert.InDeltaSlice(t, []mat.Float{2.37258}, loss.Value().Data(), 0.00001)
}

func newTestModel() *Model {
	model := New(4)
	model.TransitionScores.Value().SetData([]mat.Float{
		0.0, 0.6, 0.8, 1.2, 1.6,
		0.2, 0.5, 0.02, 0.03, 0.45,
		0.3, 0.2, 0.6, 0.01, 0.19,
		0.4, 0.02, 0.02, 0.7, 0.26,
		0.9, 0.1, 0.02, 0.08, 0.8,
	})
	return model
}
