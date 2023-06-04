// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestSoftShrink_Forward(t *testing.T) {
	t.Run("float32", testSoftShrinkForward[float32])
	t.Run("float64", testSoftShrinkForward[float64])
}

func testSoftShrinkForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewDense[T](mat.WithBacking([]T{0.1, -0.2, 0.3, 0.0, 0.6, -0.6})),
		grad:         nil,
		requiresGrad: true,
	}
	lambda := &variable{
		value:        mat.Scalar[T](0.2),
		grad:         nil,
		requiresGrad: false,
	}

	f := NewSoftShrink(x, lambda)
	assert.Equal(t, []*variable{x, lambda}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.0, 0.0, 0.1, 0, 0.4, -0.4}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{-1.0, 0.5, 0.8, 0.0, 1.0, 2.0})))
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.0, 0.0, 0.8, 0.0, 1.0, 2.0}, x.grad.Data(), 1.0e-6)
}
