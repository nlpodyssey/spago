// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package gradfn

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestSparseMax_Forward(t *testing.T) {
	t.Run("float32", testSparseMaxForward[float32])
	t.Run("float64", testSparseMaxForward[float64])
}

func testSparseMaxForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewDense[T](mat.WithBacking([]T{0.8053, 0.4594, -0.6136, -0.9460, 1.0722})),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewSparseMax(x)
	assert.Equal(t, []*variable{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.3597, 0.0138, 0.0000, 0.0000, 0.6265}, y.Data(), 1.0e-3)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{1.0, 0.5, 0.5, 0.5, 1.0, 1.0})))
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.16, -0.33, 0, 0, 0.16}, x.grad.Data(), 1.0e-2)
}

func TestSparseMaxLoss_Forward(t *testing.T) {
	t.Run("float32", testSparseMaxLossForward[float32])
	t.Run("float64", testSparseMaxLossForward[float64])
}

func testSparseMaxLossForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewDense[T](mat.WithBacking([]T{-0.3218, 0.7395, -0.2319, 0.2312, 0.7185})),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewSparseMaxLoss(x)

	y, err := f.Forward()
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{-1.3009, -0.2396, -1.2110, -0.7479, -0.2606}, y.Data(), 1.0e-2)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{0, 0, -1, 0, 0})))
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.0000, 0.5098, -1.0000, 0.0015, 0.4888}, x.grad.Data(), 1.0e-2)
}
