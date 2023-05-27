// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestConcat_Forward(t *testing.T) {
	t.Run("float32", testConcatForward[float32])
	t.Run("float64", testConcatForward[float64])
}

func testConcatForward[T float.DType](t *testing.T) {
	x1 := &variable{
		value:        mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 0.3})),
		grad:         nil,
		requiresGrad: true,
	}
	x2 := &variable{
		value:        mat.NewDense[T](mat.WithBacking([]T{0.4, 0.5, 0.6, 0.7})),
		grad:         nil,
		requiresGrad: true,
	}
	x3 := &variable{
		value:        mat.NewDense[T](mat.WithBacking([]T{0.8, 0.9})),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewConcat([]*variable{x1, x2, x3})
	assert.Equal(t, []*variable{x1, x2, x3}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0})))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{1.0, 2.0, 3.0}, x1.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{4.0, 5.0, 6.0, 7.0}, x2.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{8.0, 9.0}, x3.grad.Data(), 1.0e-6)
}
