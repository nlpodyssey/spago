// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layernormsimple

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := New()
	g := ag.NewGraph()

	// == Forward
	x1 := g.NewVariable(mat.NewVecDense([]float64{1.0, 2.0, 0.0, 4.0}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{3.0, 2.0, 1.0, 6.0}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{6.0, 2.0, 5.0, 1.0}), true)

	y := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x1, x2, x3)

	if !floats.EqualApprox(y[0].Value().Data(), []float64{-0.5070925528, 0.1690308509, -1.1832159566, 1.5212776585}, 1.0e-06) {
		t.Error("The output at position 0 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[1].Value().Data(), []float64{0.0, -0.5345224838, -1.0690449676, 1.6035674515}, 1.0e-06) {
		t.Error("The output at position 1 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[2].Value().Data(), []float64{1.2126781252, -0.7276068751, 0.7276068751, -1.2126781252}, 1.0e-06) {
		t.Error("The output at position 2 doesn't match the expected values")
	}

	// == Backward

	y[0].PropagateGrad(mat.NewVecDense([]float64{-1.0, -0.2, 0.4, 0.6}))
	y[1].PropagateGrad(mat.NewVecDense([]float64{-0.3, 0.1, 0.7, 0.9}))
	y[2].PropagateGrad(mat.NewVecDense([]float64{0.3, -0.4, 0.7, -0.8}))
	g.BackwardAll()

	if !floats.EqualApprox(x1.Grad().Data(), []float64{-0.5640800969, -0.1274975561, 0.4868088507, 0.2047688023}, 1.0e-06) {
		t.Error("The x1-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{-0.3474396144, -0.0878144080, 0.2787152951, 0.1565387274}, 1.0e-06) {
		t.Error("The x2-gradients don't match the expected values")
	}

	if !floats.EqualApprox(x3.Grad().Data(), []float64{-0.1440946948, 0.0185468419, 0.1754816581, -0.0499338051}, 1.0e-06) {
		t.Error("The x3-gradients don't match the expected values")
	}
}
