// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestConcat_Forward(t *testing.T) {
	t.Run("float32", testConcatForward[float32])
	t.Run("float64", testConcatForward[float64])
}

func testConcatForward[T mat.DType](t *testing.T) {
	x1 := &variable[T]{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3}),
		grad:         nil,
		requiresGrad: true,
	}
	x2 := &variable[T]{
		value:        mat.NewVecDense([]T{0.4, 0.5, 0.6, 0.7}),
		grad:         nil,
		requiresGrad: true,
	}
	x3 := &variable[T]{
		value:        mat.NewVecDense([]T{0.8, 0.9}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewConcat[T]([]*variable[T]{x1, x2, x3})
	y := f.Forward()

	assert.InDeltaSlice(t, []T{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}))

	assert.InDeltaSlice(t, []T{1.0, 2.0, 3.0}, x1.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{4.0, 5.0, 6.0, 7.0}, x2.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{8.0, 9.0}, x3.grad.Data(), 1.0e-6)
}
