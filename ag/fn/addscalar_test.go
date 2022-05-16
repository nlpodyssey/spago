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

func TestAddScalar_Forward(t *testing.T) {
	t.Run("float32", testAddScalarForward[float32])
	t.Run("float64", testAddScalarForward[float64])
}

func testAddScalarForward[T float.DType](t *testing.T) {
	x1 := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	x2 := &variable{
		value:        mat.NewScalar[T](1.0),
		grad:         nil,
		requiresGrad: false,
	}

	f := NewAddScalar(x1, x2)
	assert.Equal(t, []*variable{x1, x2}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{1.1, 1.2, 1.3, 1.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []T{-1.0, 0.5, 0.8, 0.0}, x1.grad.Data(), 1.0e-6)
}

func TestAddScalar_Forward2(t *testing.T) {
	t.Run("float32", testAddScalarForward2[float32])
	t.Run("float64", testAddScalarForward2[float64])
}

func testAddScalarForward2[T float.DType](t *testing.T) {
	x1 := &variable{
		value: mat.NewDense(3, 4, []T{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		}),
		grad:         nil,
		requiresGrad: true,
	}
	x2 := &variable{
		value:        mat.NewVecDense([]T{0.1}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewAddScalar(x1, x2)
	y := f.Forward()

	assert.InDeltaSlice(t, []T{
		0.2, 0.3, 0.4, 0.1,
		0.5, 0.6, -0.5, 0.8,
		-0.4, 0.9, -0.7, 0.0,
	}, y.Data(), 1.0e-6)

	f.Backward(mat.NewDense(3, 4, []T{
		-1.0, 0.5, 0.8, 0.0,
		1.0, 0.3, 0.6, 0.0,
		1.0, -0.5, -0.3, 0.0,
	}))

	assert.InDeltaSlice(t, []T{
		-1.0, 0.5, 0.8, 0.0,
		1.0, 0.3, 0.6, 0.0,
		1.0, -0.5, -0.3, 0.0,
	}, x1.grad.Data(), 1.0e-6)

	assert.InDeltaSlice(t, []T{2.4}, x2.grad.Data(), 1.0e-6)
}
