// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSubScalar_Forward(t *testing.T) {
	t.Run("float32", testSubScalarForward[float32])
	t.Run("float64", testSubScalarForward[float64])
}

func testSubScalarForward[T mat.DType](t *testing.T) {
	x1 := &variable[T]{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	x2 := &variable[T]{
		value:        mat.NewScalar[T](2.0),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewSubScalar[T](x1, x2)
	y := f.Forward()

	assert.InDeltaSlice(t, []T{-1.9, -1.8, -1.7, -2.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []T{-1.0, 0.5, 0.8, 0.0}, x1.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{-0.3}, x2.grad.Data(), 1.0e-6)
}
