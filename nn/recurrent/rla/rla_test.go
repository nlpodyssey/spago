// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rla

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_ForwardWithPrev(t *testing.T) {
	t.Run("float32", testModelForwardWithPrev[float32])
	t.Run("float64", testModelForwardWithPrev[float64])
}

func testModelForwardWithPrev[T mat.DType](t *testing.T) {
	model := newTestModel[T]()

	// == Forward
	x0 := ag.NewVariable[T](mat.NewVecDense([]T{-0.8, -0.9, -0.9, 1.0}), true)
	s1 := model.Next(nil, x0)

	assert.InDeltaSlice(t, []T{0.88, -1.1, -0.45, 0.41}, s1.Y.Value().Data(), 1.0e-05)

	x1 := ag.NewVariable[T](mat.NewVecDense([]T{0.8, -0.3, 0.5, 0.3}), true)
	s2 := model.Next(s1, x1)

	assert.InDeltaSlice(t, []T{0.5996537, -0.545537, -0.63689751, 0.453609420}, s2.Y.Value().Data(), 1.0e-05)
}

func newTestModel[T mat.DType]() *Model[T] {
	model := New[T](Config{
		InputSize: 4,
	})
	model.Wv.Value().SetData([]T{
		0.5, 0.6, -0.8, 0.7,
		-0.4, 0.1, 0.7, -0.7,
		0.3, 0.8, -0.9, 0.0,
		0.5, -0.4, -0.5, -0.3,
	})
	model.Bv.Value().SetData([]T{0.4, 0.0, -0.3, 0.3})
	model.Wk.Value().SetData([]T{
		0.7, -0.2, -0.1, 0.2,
		-0.1, -0.1, 0.3, -0.2,
		0.6, 0.1, 0.9, 0.3,
		0.3, 0.6, 0.4, 0.2,
	})
	model.Bk.Value().SetData([]T{0.8, -0.2, -0.5, -0.9})
	model.Wq.Value().SetData([]T{
		-0.8, -0.6, 0.2, 0.5,
		0.7, -0.6, -0.3, 0.6,
		-0.3, 0.3, 0.4, -0.8,
		0.8, 0.2, 0.4, 0.3,
	})
	model.Bq.Value().SetData([]T{0.3, 0.5, -0.7, -0.6})
	return model
}
