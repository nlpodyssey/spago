// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSoftPlusForward(t *testing.T) {
	t.Run("float32", testSoftPlusForward[float32])
	t.Run("float64", testSoftPlusForward[float64])
}

func testSoftPlusForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value:        mat.NewVecDense([]T{0.1, -0.2, 20.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	beta := &variable[T]{
		value:        mat.NewScalar[T](2.0),
		grad:         nil,
		requiresGrad: false,
	}
	threshold := &variable[T]{
		value:        mat.NewScalar[T](20.0),
		grad:         nil,
		requiresGrad: false,
	}

	f := NewSoftPlus[T](x, beta, threshold)
	assert.Equal(t, []*variable[T]{x, beta, threshold}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{0.399069434, 0.25650762, 20.3, 0.346573590}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []T{-0.5498339, 0.20065616, 0.8, 0}, x.grad.Data(), 1.0e-6)
}
