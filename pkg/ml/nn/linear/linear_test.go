// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linear

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModel_Forward(t *testing.T) {

	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0}), true)
	y := activation.New(ag.OpTanh).NewProc(ctx).Forward(model.NewProc(ctx).Forward(x)[0])[0] // TODO: test linear only

	if !floats.EqualApprox(y.Value().Data(), []float64{-0.39693, -0.79688, 0.0, 0.70137, -0.18775}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]float64{0.0, 0.5, -0.4, -0.9, 0.9}), false)
	loss := losses.MSE(g, y, gold, false)
	g.Backward(loss)

	if !floats.EqualApprox(x.Grad().Data(), []float64{0.0126, -2.07296, 1.07476, -0.14158}, 0.005) {
		t.Error("The input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.W.Grad().(*mat.Dense).Data(), []float64{
		0.26751, 0.30095, 0.30095, -0.33439,
		0.37867, 0.42601, 0.42601, -0.47334,
		-0.32, -0.36, -0.36, 0.4,
		-0.65089, -0.73226, -0.73226, 0.81362,
		0.83952, 0.94446, 0.94446, -1.04940,
	}, 1.0e-05) {
		t.Error("W doesn't match the expected values")
	}

	if !floats.EqualApprox(model.B.Grad().Data(), []float64{
		-0.33439, -0.47334, 0.4, 0.81362, -1.0494,
	}, 1.0e-05) {
		t.Error("B doesn't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(4, 5)
	model.W.Value().SetData([]float64{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
		0.4, 1.0, -0.7, 0.8,
	})
	model.B.Value().SetData([]float64{0.4, 0.0, -0.3, 0.8, -0.4})
	return model
}
