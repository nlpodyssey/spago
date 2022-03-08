// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestModel_SelfAttention(t *testing.T) {
	t.Run("float32", testModelSelfAttention[float32])
	t.Run("float64", testModelSelfAttention[float64])
}

func testModelSelfAttention[T mat.DType](t *testing.T) {
	model := newTestModel[T]()
	s := ag.NewSession[T](model, ag.Training)
	defer s.Close()

	x1 := s.NewVariable(mat.NewVecDense([]T{-0.8, -0.9, -0.9, 1.0}), true)
	x2 := s.NewVariable(mat.NewVecDense([]T{0.8, -0.3, 0.5, 0.3}), true)
	x3 := s.NewVariable(mat.NewVecDense([]T{-0.2, 0.7, 0.2, 0.4}), true)

	output, _, _ := s.Module().Forward(Cache[T]{}, []ag.Node[T]{x1, x2, x3})

	assert.InDeltaSlice(t, []T{0.789110, -0.755551, -0.431247}, output[0].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{0.780654, -0.6212001, -0.380214}, output[1].Value().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{0.7586521, -0.569575, -0.390976}, output[2].Value().Data(), 1.0e-05)

	output[0].PropagateGrad(mat.NewVecDense([]T{-0.04, 0.36, 0.32}))
	output[1].PropagateGrad(mat.NewVecDense([]T{-0.08, -0.2, -0.1}))
	output[2].PropagateGrad(mat.NewVecDense([]T{0.1, 0.3, 0.8}))

	s.Graph().Backward(nil)

	assert.InDeltaSlice(t, []T{0.01654154, 0.48942297, -0.1587743, -0.2387454}, x1.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{0.04408108, 0.18716132, -0.15425818, -0.040870}, x2.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{0.05800791, 0.205784865, -0.2431444, -0.1281430}, x3.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.01790323, 0.01853984, 0.01959262, -0.020789988,
		-0.1529254, -0.2133677, -0.17563661, 0.336455424,
		-0.2887047, -0.3777314, -0.31337569, 0.705232107,
	}, model.Value.W.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{
		-0.02, 0.46, 1.02,
	}, model.Value.B.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{
		0.07438275, 0.15194683, 0.11696175, -0.0629919,
		0.03235329, 0.05018469, 0.04422187, -0.0234946,
		-0.0599427, -0.1594204, -0.1097165, 0.0598379,
	}, model.Key.W.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{
		0, 0, 0,
	}, model.Key.B.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{
		0.00538138, -0.0264289, -0.0085512, -0.0088408,
		-0.0175901, -0.0032803, -0.0132455, 0.0143783,
		-0.1022365, -0.0221910, -0.0784209, 0.08306303,
	}, model.Query.W.Grad().Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{
		-0.0267918, 0.0149118, 0.08413719,
	}, model.Query.B.Grad().Data(), 1.0e-05)
}

func newTestModel[T mat.DType]() *SelfAttention[T] {
	model := &SelfAttention[T]{New(Config[T]{
		InputSize:   4,
		QuerySize:   3,
		KeySize:     3,
		ValueSize:   3,
		ScaleFactor: 1.0 / mat.Sqrt[T](3.0),
	})}

	model.Value.W.Value().SetData([]T{
		0.5, 0.6, -0.8, 0.7,
		-0.4, 0.1, 0.7, -0.7,
		0.3, 0.8, -0.9, 0.0,
	})
	model.Value.B.Value().SetData([]T{0.4, 0.0, -0.3})
	model.Key.W.Value().SetData([]T{
		0.7, -0.2, -0.1, 0.2,
		-0.1, -0.1, 0.3, -0.2,
		0.6, 0.1, 0.9, 0.3,
	})
	model.Key.B.Value().SetData([]T{0.8, -0.2, -0.5})
	model.Query.W.Value().SetData([]T{
		-0.8, -0.6, 0.2, 0.5,
		0.7, -0.6, -0.3, 0.6,
		-0.3, 0.3, 0.4, -0.8,
	})
	model.Query.B.Value().SetData([]T{0.3, 0.5, -0.7})
	return model
}
