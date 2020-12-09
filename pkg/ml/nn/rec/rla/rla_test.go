// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rla

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModel_ForwardWithPrev(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	proc := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)

	// == Forward
	x0 := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	_ = proc.Forward(x0)
	s0 := proc.LastState()

	if !floats.EqualApprox(s0.Y.Value().Data(), []float64{0.88, -1.1, -0.45, 0.41}, 1.0e-05) {
		t.Error("The output 0 doesn't match the expected values")
	}

	x1 := g.NewVariable(mat.NewVecDense([]float64{0.8, -0.3, 0.5, 0.3}), true)
	_ = proc.Forward(x1)
	s1 := proc.LastState()

	if !floats.EqualApprox(s1.Y.Value().Data(), []float64{0.5996537, -0.545537, -0.63689751, 0.453609420}, 1.0e-05) {
		t.Error("The output 1 doesn't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(Config{
		InputSize: 4,
	})
	model.Wv.Value().SetData([]float64{
		0.5, 0.6, -0.8, 0.7,
		-0.4, 0.1, 0.7, -0.7,
		0.3, 0.8, -0.9, 0.0,
		0.5, -0.4, -0.5, -0.3,
	})
	model.Bv.Value().SetData([]float64{0.4, 0.0, -0.3, 0.3})
	model.Wk.Value().SetData([]float64{
		0.7, -0.2, -0.1, 0.2,
		-0.1, -0.1, 0.3, -0.2,
		0.6, 0.1, 0.9, 0.3,
		0.3, 0.6, 0.4, 0.2,
	})
	model.Bk.Value().SetData([]float64{0.8, -0.2, -0.5, -0.9})
	model.Wq.Value().SetData([]float64{
		-0.8, -0.6, 0.2, 0.5,
		0.7, -0.6, -0.3, 0.6,
		-0.3, 0.3, 0.4, -0.8,
		0.8, 0.2, 0.4, 0.3,
	})
	model.Bq.Value().SetData([]float64{0.3, 0.5, -0.7, -0.6})
	return model
}
