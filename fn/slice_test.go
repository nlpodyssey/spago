// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestSliceForward(t *testing.T) {
	t.Run("float32", testSliceForward[float32])
	t.Run("float64", testSliceForward[float64])
}

func testSliceForward[T float.DType](t *testing.T) {
	x := &variable{
		value: mat.NewDense(3, 4, []T{
			11, 12, 13, 14,
			21, 22, 23, 24,
			31, 32, 33, 34,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewSlice(x, 1, 1, 3, 3)
	assert.Equal(t, []*variable{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)

	mat.AssertMatrixEquals(t, mat.NewDense(2, 2, []T{
		22, 23,
		32, 33,
	}), y)

	err = f.Backward(mat.NewDense(2, 2, []T{
		1, 2,
		3, 4,
	}))
	assert.Nil(t, err)

	mat.AssertMatrixEquals(t, mat.NewDense(3, 4, []T{
		0, 0, 0, 0,
		0, 1, 2, 0,
		0, 3, 4, 0,
	}), x.grad)
}
