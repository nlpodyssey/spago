// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package birnn

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/recurrent/srn"
	"github.com/stretchr/testify/assert"
)

func TestModelConcat_Forward(t *testing.T) {
	t.Run("float32", testModelConcatForward[float32])
	t.Run("float64", testModelConcatForward[float64])
}

func testModelConcatForward[T float.DType](t *testing.T) {
	model := newTestModel[T](Concat)

	// == Forward

	x1 := mat.NewVecDense([]T{0.5, 0.6}, mat.WithGrad(true))
	x2 := mat.NewVecDense([]T{0.7, -0.4}, mat.WithGrad(true))
	x3 := mat.NewVecDense([]T{0.0, -0.7}, mat.WithGrad(true))
	y := model.Forward(x1, x2, x3)

	assert.InDeltaSlice(t, []T{
		0.187746, -0.50052, 0.109558, -0.005277, -0.084306, -0.628766,
	}, y[0].Value().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []T{
		-0.704648, 0.200908, -0.064056, -0.329084, -0.237601, -0.449676,
	}, y[1].Value().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []T{
		0.256521, 0.725227, 0.781582, 0.129273, -0.716298, -0.263625,
	}, y[2].Value().Data(), 1.0e-06)

	// == Backward

	y[0].AccGrad(mat.NewVecDense([]T{-0.4, -0.8, 0.1, 0.4, 0.6, -0.4}))
	y[1].AccGrad(mat.NewVecDense([]T{0.6, 0.6, 0.7, 0.7, -0.6, 0.3}))
	y[2].AccGrad(mat.NewVecDense([]T{-0.1, -0.1, 0.1, -0.8, 0.4, -0.5}))

	ag.Backward(y...)

	// Important! average params by sequence length
	nn.ForEachParam(model, func(param nn.Param) {
		param.Grad().ProdScalarInPlace(1.0 / 3.0)
	})

	assert.InDeltaSlice(t, []T{1.031472, -0.627913}, x1.Grad().Data(), 0.006)
	assert.InDeltaSlice(t, []T{-0.539497, -0.629167}, x2.Grad().Data(), 0.006)
	assert.InDeltaSlice(t, []T{0.013097, -0.09932}, x3.Grad().Data(), 0.006)

	assert.InDeltaSlice(t, []T{
		0.001234, -0.107987,
		0.175039, 0.015738,
		0.213397, -0.046717,
	}, model.Positive.(*srn.Model).W.Grad().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []T{
		0.041817, -0.059241, 0.013592,
		0.042229, -0.086071, 0.019157,
		0.035331, -0.11595, 0.02512,
	}, model.Positive.(*srn.Model).WRec.Grad().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []T{
		-0.071016, 0.268027, 0.345019,
	}, model.Positive.(*srn.Model).B.Grad().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []T{
		0.145713, 0.234548,
		0.050135, 0.070768,
		-0.06125, -0.017281,
	}, model.Negative.(*srn.Model).W.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.029278, -0.112568, -0.089725,
		-0.074426, 0.003116, -0.070784,
		0.022664, 0.040583, 0.044139,
	}, model.Negative.(*srn.Model).WRec.Grad().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []T{
		-0.03906, 0.237598, -0.137858,
	}, model.Negative.(*srn.Model).B.Grad().Data(), 1.0e-06)
}

func TestModelSum_Forward(t *testing.T) {
	t.Run("float32", testModelSumForward[float32])
	t.Run("float64", testModelSumForward[float64])
}

func testModelSumForward[T float.DType](t *testing.T) {
	model := newTestModel[T](Sum)

	// == Forward

	x1 := mat.NewVecDense([]T{0.5, 0.6}, mat.WithGrad(true))
	x2 := mat.NewVecDense([]T{0.7, -0.4}, mat.WithGrad(true))
	x3 := mat.NewVecDense([]T{0.0, -0.7}, mat.WithGrad(true))
	y := model.Forward(x1, x2, x3)

	assert.InDeltaSlice(t, []T{0.182469, -0.584826, -0.519207}, y[0].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{-1.033731, -0.036692, -0.513732}, y[1].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{0.385793, 0.008929, 0.517957}, y[2].Value().Data(), 1.0e-06)
}

func TestModelAvg_Forward(t *testing.T) {
	t.Run("float32", testModelAvgForward[float32])
	t.Run("float64", testModelAvgForward[float64])
}

func testModelAvgForward[T float.DType](t *testing.T) {
	model := newTestModel[T](Avg)

	// == Forward

	x1 := mat.NewVecDense([]T{0.5, 0.6}, mat.WithGrad(true))
	x2 := mat.NewVecDense([]T{0.7, -0.4}, mat.WithGrad(true))
	x3 := mat.NewVecDense([]T{0.0, -0.7}, mat.WithGrad(true))
	y := model.Forward(x1, x2, x3)

	assert.InDeltaSlice(t, []T{0.0912345, -0.292413, -0.2596035}, y[0].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{-0.5168655, -0.018346, -0.256866}, y[1].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{0.1928965, 0.0044645, 0.2589785}, y[2].Value().Data(), 1.0e-06)
}

func TestModelProd_Forward(t *testing.T) {
	t.Run("float32", testModelProdForward[float32])
	t.Run("float64", testModelProdForward[float64])
}

func testModelProdForward[T float.DType](t *testing.T) {
	model := newTestModel[T](Prod)

	// == Forward

	x1 := mat.NewVecDense([]T{0.5, 0.6}, mat.WithGrad(true))
	x2 := mat.NewVecDense([]T{0.7, -0.4}, mat.WithGrad(true))
	x3 := mat.NewVecDense([]T{0.0, -0.7}, mat.WithGrad(true))
	y := model.Forward(x1, x2, x3)

	assert.InDeltaSlice(t, []T{-0.00099, 0.042197, -0.068886}, y[0].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{0.231888, -0.047735, 0.028804}, y[1].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{0.033161, -0.519478, -0.206044}, y[2].Value().Data(), 1.0e-06)
}

func newTestModel[T float.DType](mergeType MergeType) *Model {
	model := New(
		srn.New[T](2, 3),
		srn.New[T](2, 3),
		mergeType,
	)
	initPos[T](model.Positive.(*srn.Model))
	initNeg[T](model.Negative.(*srn.Model))
	return model
}

func initPos[T float.DType](m *srn.Model) {
	mat.SetData[T](m.W.Value(), []T{
		-0.9, 0.4,
		0.7, -1.0,
		-0.9, -0.4,
	})
	mat.SetData[T](m.WRec.Value(), []T{
		0.1, 0.9, -0.5,
		-0.6, 0.7, 0.7,
		0.3, 0.9, 0.0,
	})
	mat.SetData[T](m.B.Value(), []T{0.4, -0.3, 0.8})
}

func initNeg[T float.DType](m *srn.Model) {
	mat.SetData[T](m.W.Value(), []T{
		0.3, 0.1,
		0.6, 0.0,
		-0.7, 0.1,
	})
	mat.SetData[T](m.WRec.Value(), []T{
		-0.2, 0.7, 0.7,
		-0.2, 0.0, -1.0,
		0.5, -0.4, 0.4,
	})
	mat.SetData[T](m.B.Value(), []T{0.2, -0.9, -0.2})
}
