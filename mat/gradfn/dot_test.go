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

func TestDot_Forward(t *testing.T) {
	t.Run("float32", testDotForward[float32])
	t.Run("float64", testDotForward[float64])
}

func testDotForward[T float.DType](t *testing.T) {

	x1 := mat.NewDense[T](mat.WithShape(3, 4), mat.WithBacking([]T{
		0.1, 0.2, 0.3, 0.0,
		0.4, 0.5, -0.6, 0.7,
		-0.5, 0.8, -0.8, -0.1,
	}), mat.WithGrad(true))

	x2 := mat.NewDense[T](mat.WithShape(3, 4), mat.WithBacking([]T{
		0.1, 0.8, 0.3, 0.1,
		0.1, -0.5, -0.9, 0.2,
		-0.2, 0.3, -0.4, -0.5,
	}), mat.WithGrad(true))

	f := NewDot(x1, x2)
	assert.Equal(t, []mat.Tensor{x1, x2}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{1.44}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{0.5})))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{
		0.05, 0.4, 0.15, 0.05,
		0.05, -0.25, -0.45, 0.1,
		-0.1, 0.15, -0.2, -0.25,
	}, x1.Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{
		0.05, 0.1, 0.15, 0.0,
		0.2, 0.25, -0.3, 0.35,
		-0.25, 0.4, -0.4, -0.05,
	}, x2.Grad().Data(), 1.0e-6)
}
