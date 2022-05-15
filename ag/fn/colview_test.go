// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestCol_Forward(t *testing.T) {
	t.Run("float32", testColForward[float32])
	t.Run("float64", testColForward[float64])
}

func testColForward[T mat.DType](t *testing.T) {
	x := &variable{
		value: mat.NewDense(3, 4, []T{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewColView(x, 2)
	assert.Equal(t, []*variable{x}, f.Operands())

	y := f.Forward()

	assert.InDeltaSlice(t, []T{
		0.3, -0.6, -0.8,
	}, y.Data(), 1.0e-6)

	if y.Rows() != 3 || y.Columns() != 1 {
		t.Error("The rows and columns of the resulting matrix are not correct")
	}

	f.Backward(mat.NewDense(3, 1, []T{
		0.1, 0.2, -0.8,
	}))

	assert.InDeltaSlice(t, []T{
		0.0, 0.0, 0.1, 0.0,
		0.0, 0.0, 0.2, 0.0,
		0.0, 0.0, -0.8, 0.0,
	}, x.grad.Data(), 1.0e-6)
}
