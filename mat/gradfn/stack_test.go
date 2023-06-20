// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"github.com/stretchr/testify/assert"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

func TestStack_Forward(t *testing.T) {
	t.Run("float32", testStackForward[float32])
	t.Run("float64", testStackForward[float64])
}

func testStackForward[T float.DType](t *testing.T) {
	x1 := mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 0.3, 0.5}), mat.WithGrad(true))
	x2 := mat.NewDense[T](mat.WithBacking([]T{0.4, 0.5, 0.6, 0.4}), mat.WithGrad(true))
	x3 := mat.NewDense[T](mat.WithBacking([]T{0.8, 0.9, 0.7, 0.6}), mat.WithGrad(true))

	f := NewStack([]mat.Tensor{x1, x2, x3})
	assert.Equal(t, []mat.Tensor{x1, x2, x3}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.1, 0.2, 0.3, 0.5, 0.4, 0.5, 0.6, 0.4, 0.8, 0.9, 0.7, 0.6}, y.Data(), 1.0e-6)

	if y.Shape()[0] != 3 && y.Shape()[1] != 4 {
		t.Error("The output size doesn't match the expected values")
	}

	err = f.Backward(mat.NewDense[T](mat.WithShape(3, 4), mat.WithBacking([]T{
		1.0, 2.0, 3.0, 4.0,
		4.0, 5.0, 6.0, 0.5,
		7.0, 8.0, 9.0, -0.3,
	})))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{1.0, 2.0, 3.0, 4.0}, x1.Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{4.0, 5.0, 6.0, 0.5}, x2.Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{7.0, 8.0, 9.0, -0.3}, x3.Grad().Data(), 1.0e-6)
}
