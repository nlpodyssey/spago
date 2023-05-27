// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fixnorm

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestModel_Forward(t *testing.T) {
	t.Run("float32", testModelForward[float32])
	t.Run("float64", testModelForward[float64])
}

func testModelForward[T float.DType](t *testing.T) {
	model := New()
	// == Forward
	x1 := mat.NewDense[T](mat.WithBacking([]T{1.0, 2.0, 0.0, 4.0}), mat.WithGrad(true))
	x2 := mat.NewDense[T](mat.WithBacking([]T{3.0, 2.0, 1.0, 6.0}), mat.WithGrad(true))
	x3 := mat.NewDense[T](mat.WithBacking([]T{6.0, 2.0, 5.0, 1.0}), mat.WithGrad(true))
	y := model.Forward(x1, x2, x3)

	assert.InDeltaSlice(t, []T{0.2182178902, 0.4364357805, 0.0, 0.8728715609}, y[0].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{0.4242640687, 0.2828427125, 0.1414213562, 0.8485281374}, y[1].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{0.7385489459, 0.246182982, 0.6154574549, 0.123091491}, y[2].Value().Data(), 1.0e-06)

	// == Backward
	y[0].AccGrad(mat.NewDense[T](mat.WithBacking([]T{-1.0, -0.2, 0.4, 0.6})))
	y[1].AccGrad(mat.NewDense[T](mat.WithBacking([]T{-0.3, 0.1, 0.7, 0.9})))
	y[2].AccGrad(mat.NewDense[T](mat.WithBacking([]T{0.3, -0.4, 0.7, -0.8})))
	ag.Backward(y...)

	assert.InDeltaSlice(t, []T{-0.2286092183, -0.0644262343, 0.0872871561, 0.0893654217}, x1.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{-0.0882469263, -0.0164048773, 0.0837214429, 0.0356381818}, x2.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{-0.0044760542, -0.0630377636, 0.0516611258, -0.1053737764}, x3.Grad().Data(), 1.0e-06)
}
