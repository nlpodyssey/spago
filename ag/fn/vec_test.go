// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestVec_Forward(t *testing.T) {
	t.Run("float32", testVecForward[float32])
	t.Run("float64", testVecForward[float64])
}

func testVecForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value: mat.NewDense(3, 4, []T{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewVec[T](x)
	y := f.Forward()

	assert.InDeltaSlice(t, []T{
		0.1, 0.2, 0.3,
		0.0, 0.4, 0.5,
		-0.6, 0.7, -0.5,
		0.8, -0.8, -0.1,
	}, y.Data(), 1.0e-6)

	if y.Rows() != 12 || y.Columns() != 1 {
		t.Error("The rows and columns of the resulting matrix are not correct")
	}

	f.Backward(mat.NewVecDense([]T{
		0.1, 0.2, 0.3,
		0.0, 0.4, 0.5,
		-0.6, 0.7, -0.5,
		0.8, -0.8, -0.1}))

	assert.InDeltaSlice(t, []T{
		0.1, 0.2, 0.3,
		0.0, 0.4, 0.5,
		-0.6, 0.7, -0.5,
		0.8, -0.8, -0.1,
	}, x.grad.Data(), 1.0e-6)

	if x.grad.Rows() != 3 || x.grad.Columns() != 4 {
		t.Error("The rows and columns of the resulting x-gradients matrix are not correct")
	}
}