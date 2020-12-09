// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package birnn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/rec/srn"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModelConcat_Forward(t *testing.T) {
	model := newTestModel(Concat)
	g := ag.NewGraph()

	// == Forward

	x1 := g.NewVariable(mat.NewVecDense([]float64{0.5, 0.6}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{0.7, -0.4}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{0.0, -0.7}), true)

	y := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x1, x2, x3)

	if !floats.EqualApprox(y[0].Value().Data(), []float64{
		0.187746, -0.50052, 0.109558, -0.005277, -0.084306, -0.628766,
	}, 1.0e-06) {
		t.Error("The first output doesn't match the expected values")
	}

	if !floats.EqualApprox(y[1].Value().Data(), []float64{
		-0.704648, 0.200908, -0.064056, -0.329084, -0.237601, -0.449676,
	}, 1.0e-06) {
		t.Error("The first output doesn't match the expected values")
	}

	if !floats.EqualApprox(y[2].Value().Data(), []float64{
		0.256521, 0.725227, 0.781582, 0.129273, -0.716298, -0.263625,
	}, 1.0e-06) {
		t.Error("The third output doesn't match the expected values")
	}

	// == Backward

	y[0].PropagateGrad(mat.NewVecDense([]float64{-0.4, -0.8, 0.1, 0.4, 0.6, -0.4}))
	y[1].PropagateGrad(mat.NewVecDense([]float64{0.6, 0.6, 0.7, 0.7, -0.6, 0.3}))
	y[2].PropagateGrad(mat.NewVecDense([]float64{-0.1, -0.1, 0.1, -0.8, 0.4, -0.5}))

	g.BackwardAll()

	// Important! average params by sequence length
	nn.ForEachParam(model, func(param *nn.Param) {
		param.Grad().ProdScalarInPlace(1.0 / 3.0)
	})

	if !floats.EqualApprox(x1.Grad().Data(), []float64{1.031472, -0.627913}, 0.006) {
		t.Error("The first input gradients don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{-0.539497, -0.629167}, 0.006) {
		t.Error("The second input gradients don't match the expected values")
	}

	if !floats.EqualApprox(x3.Grad().Data(), []float64{0.013097, -0.09932}, 0.006) {
		t.Error("The third input gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.Positive.(*srn.Model).W.Grad().(*mat.Dense).Data(), []float64{
		0.001234, -0.107987,
		0.175039, 0.015738,
		0.213397, -0.046717,
	}, 1.0e-06) {
		t.Error("W-positive doesn't match the expected values")
	}

	if !floats.EqualApprox(model.Positive.(*srn.Model).WRec.Grad().(*mat.Dense).Data(), []float64{
		0.041817, -0.059241, 0.013592,
		0.042229, -0.086071, 0.019157,
		0.035331, -0.11595, 0.02512,
	}, 1.0e-06) {
		t.Error("WRec-positive doesn't match the expected values")
	}

	if !floats.EqualApprox(model.Positive.(*srn.Model).B.Grad().Data(), []float64{
		-0.071016, 0.268027, 0.345019,
	}, 1.0e-06) {
		t.Error("B-positive doesn't match the expected values")
	}

	if !floats.EqualApprox(model.Negative.(*srn.Model).W.Grad().(*mat.Dense).Data(), []float64{
		0.145713, 0.234548,
		0.050135, 0.070768,
		-0.06125, -0.017281,
	}, 1.0e-05) {
		t.Error("W-negative doesn't match the expected values")
	}

	if !floats.EqualApprox(model.Negative.(*srn.Model).WRec.Grad().(*mat.Dense).Data(), []float64{
		-0.029278, -0.112568, -0.089725,
		-0.074426, 0.003116, -0.070784,
		0.022664, 0.040583, 0.044139,
	}, 1.0e-06) {
		t.Error("WRec-negative doesn't match the expected values")
	}

	if !floats.EqualApprox(model.Negative.(*srn.Model).B.Grad().Data(), []float64{
		-0.03906, 0.237598, -0.137858,
	}, 1.0e-06) {
		t.Error("B-negative doesn't match the expected values")
	}
}

func TestModelSum_Forward(t *testing.T) {
	model := newTestModel(Sum)
	g := ag.NewGraph()

	// == Forward

	x1 := g.NewVariable(mat.NewVecDense([]float64{0.5, 0.6}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{0.7, -0.4}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{0.0, -0.7}), true)

	y := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x1, x2, x3)

	if !floats.EqualApprox(y[0].Value().Data(), []float64{0.182469, -0.584826, -0.519207}, 1.0e-06) {
		t.Error("The first output doesn't match the expected values")
	}

	if !floats.EqualApprox(y[1].Value().Data(), []float64{-1.033731, -0.036692, -0.513732}, 1.0e-06) {
		t.Error("The second output doesn't match the expected values")
	}

	if !floats.EqualApprox(y[2].Value().Data(), []float64{0.385793, 0.008929, 0.517957}, 1.0e-06) {
		t.Error("The third output doesn't match the expected values")
	}
}

func TestModelAvg_Forward(t *testing.T) {
	model := newTestModel(Avg)
	g := ag.NewGraph()

	// == Forward

	x1 := g.NewVariable(mat.NewVecDense([]float64{0.5, 0.6}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{0.7, -0.4}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{0.0, -0.7}), true)

	y := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x1, x2, x3)

	if !floats.EqualApprox(y[0].Value().Data(), []float64{0.0912345, -0.292413, -0.2596035}, 1.0e-06) {
		t.Error("The first output doesn't match the expected values")
	}

	if !floats.EqualApprox(y[1].Value().Data(), []float64{-0.5168655, -0.018346, -0.256866}, 1.0e-06) {
		t.Error("The second output doesn't match the expected values")
	}

	if !floats.EqualApprox(y[2].Value().Data(), []float64{0.1928965, 0.0044645, 0.2589785}, 1.0e-06) {
		t.Error("The third output doesn't match the expected values")
	}
}

func TestModelProd_Forward(t *testing.T) {
	model := newTestModel(Prod)
	g := ag.NewGraph()

	// == Forward

	x1 := g.NewVariable(mat.NewVecDense([]float64{0.5, 0.6}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{0.7, -0.4}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{0.0, -0.7}), true)

	y := model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).Forward(x1, x2, x3)

	if !floats.EqualApprox(y[0].Value().Data(), []float64{-0.00099, 0.042197, -0.068886}, 1.0e-06) {
		t.Error("The first output doesn't match the expected values")
	}

	if !floats.EqualApprox(y[1].Value().Data(), []float64{0.231888, -0.047735, 0.028804}, 1.0e-06) {
		t.Error("The second output doesn't match the expected values")
	}

	if !floats.EqualApprox(y[2].Value().Data(), []float64{0.033161, -0.519478, -0.206044}, 1.0e-06) {
		t.Error("The third output doesn't match the expected values")
	}
}

func newTestModel(mergeType MergeType) *Model {
	model := New(
		srn.New(2, 3),
		srn.New(2, 3),
		mergeType,
	)
	initPos(model.Positive.(*srn.Model))
	initNeg(model.Negative.(*srn.Model))
	return model
}

func initPos(m *srn.Model) {
	m.W.Value().SetData([]float64{
		-0.9, 0.4,
		0.7, -1.0,
		-0.9, -0.4,
	})
	m.WRec.Value().SetData([]float64{
		0.1, 0.9, -0.5,
		-0.6, 0.7, 0.7,
		0.3, 0.9, 0.0,
	})
	m.B.Value().SetData([]float64{0.4, -0.3, 0.8})
}

func initNeg(m *srn.Model) {
	m.W.Value().SetData([]float64{
		0.3, 0.1,
		0.6, 0.0,
		-0.7, 0.1,
	})
	m.WRec.Value().SetData([]float64{
		-0.2, 0.7, 0.7,
		-0.2, 0.0, -1.0,
		0.5, -0.4, 0.4,
	})
	m.B.Value().SetData([]float64{0.2, -0.9, -0.2})
}
