// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gru

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/losses"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestModel_Forward(t *testing.T) {
	t.Run("float32", testModelForward[float32])
	t.Run("float64", testModelForward[float64])
}

func testModelForward[T mat.DType](t *testing.T) {
	model := newTestModel[T]()

	// == Forward

	x := ag.NewVariable[T](mat.NewVecDense([]T{-0.8, -0.9, -0.9, 1.0}), true)
	y := model.Forward(x)[0]

	assert.InDeltaSlice(t, []T{0.74, -0.23, 0.11, 0.49, -0.05}, y.Value().Data(), 0.005)

	// == Backward

	gold := ag.NewVariable[T](mat.NewVecDense([]T{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(y, gold, false)
	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.53, -0.49, 0.18, 0.20}, x.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		-0.01, -0.02, -0.02, 0.02,
		-0.10, -0.12, -0.12, 0.13,
		-0.02, -0.02, -0.02, 0.03,
		0.22, 0.24, 0.24, -0.27,
		-0.02, -0.02, -0.02, 0.02,
	}, model.WPart.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{0.02, 0.13, 0.03, -0.27, 0.02}, model.BPart.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		-0.03, -0.03, -0.03, 0.04,
		0.24, 0.27, 0.27, -0.30,
		0.00, 0.00, 0.00, 0.00,
		0.06, 0.06, 0.06, -0.07,
		0.09, 0.10, 0.10, -0.12,
	}, model.WCand.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{0.04, -0.3, 0.0, -0.07, -0.12}, model.BCand.Grad().Data(), 0.005)

	if model.BRes.HasGrad() {
		t.Error("BRes doesn't match the expected values")
	}

	if model.WRes.HasGrad() {
		t.Error("WRes doesn't match the expected values")
	}

	if model.WPartRec.HasGrad() {
		t.Error("WPartRec doesn't match the expected values")
	}

	if model.WResRec.HasGrad() {
		t.Error("WResRec doesn't match the expected values")
	}

	if model.WCandRec.HasGrad() {
		t.Error("WCandRec doesn't match the expected values")
	}
}

func TestModel_ForwardWithPrev(t *testing.T) {
	t.Run("float32", testModelForwardWithPrev[float32])
	t.Run("float64", testModelForwardWithPrev[float64])
}

func testModelForwardWithPrev[T mat.DType](t *testing.T) {
	model := newTestModel[T]()

	// == Forward

	s0 := &State[T]{Y: ag.NewVariable[T](mat.NewVecDense([]T{-0.2, 0.2, -0.3, -0.9, -0.8}), true)}
	x := ag.NewVariable[T](mat.NewVecDense([]T{-0.8, -0.9, -0.9, 1.0}), true)
	s1 := model.Next(s0, x)

	assert.InDeltaSlice(t, []T{0.86, 0.18, -0.24, 0.36, -0.36}, s1.Y.Value().Data(), 0.005)

	// == Backward

	gold := ag.NewVariable[T](mat.NewVecDense([]T{0.57, 0.75, -0.15, 1.64, 0.45}), false)
	loss := losses.MSE(s1.Y, gold, false)
	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.56, -0.83, 0.5, 0.55}, x.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		-0.02, -0.02, -0.02, 0.03,
		-0.01, -0.01, -0.01, 0.01,
		0.0, 0.0, 0.0, -0.01,
		0.42, 0.47, 0.47, -0.52,
		0.17, 0.2, 0.2, -0.22,
	}, model.WPart.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{0.03, 0.01, -0.01, -0.52, -0.22}, model.BPart.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		-0.02, -0.02, -0.02, 0.02,
		0.08, 0.09, 0.09, -0.10,
		0.00, 0.00, 0.00, 0.00,
		0.05, 0.05, 0.05, -0.06,
		0.22, 0.25, 0.25, -0.28,
	}, model.WCand.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{0.02, -0.1, 0.0, -0.06, -0.28}, model.BCand.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		0.0, 0.01, 0.01, -0.01,
		0.0, 0.0, 0.0, 0.0,
		0.01, 0.01, 0.01, -0.02,
		0.02, 0.02, 0.02, -0.03,
		0.01, 0.01, 0.01, -0.01,
	}, model.WRes.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{-0.01, 0.0, -0.02, -0.03, -0.01}, model.BRes.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		-0.01, 0.01, -0.01, -0.02, -0.02,
		0.0, 0.0, 0.0, -0.01, -0.01,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.1, -0.1, 0.16, 0.47, 0.42,
		0.04, -0.04, 0.07, 0.2, 0.17,
	}, model.WPartRec.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		0.0, 0.0, 0.0, 0.01, 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.01, 0.01,
		0.01, -0.01, 0.01, 0.02, 0.02,
		0.0, 0.0, 0.0, 0.01, 0.01,
	}, model.WResRec.Grad().Data(), 0.005)

	assert.InDeltaSlice(t, []T{
		0.0, 0.0, 0.0, -0.01, -0.01,
		0.01, -0.01, 0.02, 0.08, 0.04,
		0.0, 0.0, 0.0, 0.0, 0.0,
		0.01, 0.0, 0.01, 0.04, 0.02,
		0.04, -0.01, 0.05, 0.21, 0.12,
	}, model.WCandRec.Grad().Data(), 0.005)
}

func newTestModel[T mat.DType]() *Model[T] {
	params := New[T](4, 5)

	mat.SetData[T](params.WPart.Value(), []T{
		0.1, 0.4, -1.0, 0.4,
		0.7, -0.2, 0.1, 0.0,
		0.7, 0.8, -0.5, -0.3,
		-0.9, 0.9, -0.3, -0.3,
		-0.7, 0.6, -0.6, -0.8,
	})

	mat.SetData[T](params.WPartRec.Value(), []T{
		0.1, -0.6, -1.0, -0.1, -0.4,
		0.5, -0.9, 0.0, 0.8, 0.3,
		-0.3, -0.9, 0.3, 1.0, -0.2,
		0.7, 0.2, 0.3, -0.4, -0.6,
		-0.2, 0.5, -0.2, -0.9, 0.4,
	})

	mat.SetData[T](params.BPart.Value(), []T{0.9, 0.2, -0.9, 0.2, -0.9})

	mat.SetData[T](params.WRes.Value(), []T{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
		0.4, 1.0, -0.7, 0.8,
	})

	mat.SetData[T](params.WResRec.Value(), []T{
		0.0, 0.8, 0.8, -1.0, -0.7,
		-0.7, -0.8, 0.2, -0.7, 0.7,
		-0.9, 0.9, 0.7, -0.5, 0.5,
		0.0, -0.1, 0.5, -0.2, -0.8,
		-0.6, 0.6, 0.8, -0.1, -0.3,
	})

	mat.SetData[T](params.BRes.Value(), []T{0.4, 0.0, -0.3, 0.8, -0.4})

	mat.SetData[T](params.WCand.Value(), []T{
		-1.0, 0.2, 0.0, 0.2,
		-0.7, 0.7, -0.3, -0.3,
		0.3, -0.6, 0.0, 0.7,
		-1.0, -0.6, 0.9, 0.8,
		0.5, 0.8, -0.9, -0.8,
	})

	mat.SetData[T](params.WCandRec.Value(), []T{
		0.2, -0.3, -0.3, -0.5, -0.7,
		0.4, -0.1, -0.6, -0.4, -0.8,
		0.6, 0.6, 0.1, 0.7, -0.4,
		-0.8, 0.9, 0.1, -0.1, -0.2,
		-0.5, -0.3, -0.6, -0.6, 0.1,
	})

	mat.SetData[T](params.BCand.Value(), []T{0.5, -0.5, 1.0, 0.4, 0.9})

	return params
}

func TestModel_ForwardSeq(t *testing.T) {
	t.Run("float32", testModelForwardSeq[float32])
	t.Run("float64", testModelForwardSeq[float64])
}

func testModelForwardSeq[T mat.DType](t *testing.T) {
	model := newTestModel2[T]()

	// == Forward

	s0 := &State[T]{Y: ag.NewVariable[T](mat.NewVecDense([]T{0.0, 0.0}), true)}
	x := ag.NewVariable[T](mat.NewVecDense([]T{3.5, 4.0, -0.1}), true)
	s1 := model.Next(s0, x)

	assert.InDeltaSlice(t, []T{-0.634733134450701, 0.896135841414256}, s1.Y.Value().Data(), 1.0e-05)

	x2 := ag.NewVariable[T](mat.NewVecDense([]T{3.3, -2.0, 0.1}), true)
	s2 := model.Next(s1, x2)

	assert.InDeltaSlice(t, []T{0.646126994447876, 0.537141024639326}, s2.Y.Value().Data(), 1.0e-05)

	// == Backward

	s1.Y.AccGrad(mat.NewVecDense([]T{-0.052008468343874, 0.416067746750988}))
	s2.Y.AccGrad(mat.NewVecDense([]T{-0.041704888674704, 0.333639109397627}))

	ag.Backward(s2.Y)

	assert.InDeltaSlice(t, []T{0.022626682234541, 0.019282896989004, -0.05477940973827}, x.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{0.047347465801696, 0.102160284950441, -0.023609485283631}, x2.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.013817689446019, 0.037543919616777, -0.001400663526169,
		0.016466177962923, 0.310541096478206, -0.010290827228346,
	}, model.WPart.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.004475986168291, 0.001816279627817,
	}, model.BPart.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.008595802515716, 0.005209577282252, -0.000260478864113,
		-0.006145984396192, 0.003724839027995, -0.0001862419514,
	}, model.WRes.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.002604788641126, -0.001862419513998,
	}, model.BRes.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.167811472603026, -0.159044185005926, 0.003692462934823,
		0.590071690390826, -0.35441204273124, 0.017772996392842,
	}, model.WCand.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.048270296961236, 0.17877784905402,
	}, model.BCand.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.001653345658763, -0.002334244460622,
		0.001182139375782, -0.001668980878242,
	}, model.WResRec.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.005865766116558, -0.008281469753348,
		0.032083218683093, -0.045296078949356,
	}, model.WPartRec.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.001568934580973, -0.004540982737526,
		-0.049299649456851, 0.142688458693303,
	}, model.WCandRec.Grad().Data(), 1.0e-05)
}

func newTestModel2[T mat.DType]() *Model[T] {
	model := New[T](3, 2)
	mat.SetData[T](model.WRes.Value(), []T{
		-0.2, -0.3, 0.5,
		0.8, 0.2, 0.01,
	})
	mat.SetData[T](model.WResRec.Value(), []T{
		0.5, 0.3,
		0.2, -0.1,
	})
	mat.SetData[T](model.BRes.Value(), []T{-0.2, 0.1})
	mat.SetData[T](model.WPart.Value(), []T{
		0.3, 0.2, -0.4,
		0.4, 0.1, -0.6,
	})
	mat.SetData[T](model.WPartRec.Value(), []T{
		-0.5, 0.22,
		0.8, -0.6,
	})
	mat.SetData[T](model.BPart.Value(), []T{0.5, 0.3})
	mat.SetData[T](model.WCand.Value(), []T{
		-0.001, -0.3, 0.5,
		0.4, 0.6, -0.3,
	})
	mat.SetData[T](model.WCandRec.Value(), []T{
		0.2, 0.7,
		0.1, -0.1,
	})
	mat.SetData[T](model.BCand.Value(), []T{0.4, 0.3})
	return model
}
