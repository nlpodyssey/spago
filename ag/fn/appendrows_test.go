// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestAppendRowsForward(t *testing.T) {
	t.Run("float32", testAppendRowsForward[float32])
	t.Run("float64", testAppendRowsForward[float64])
}

func testAppendRowsForward[T mat.DType](t *testing.T) {
	x := &variable[T]{
		value: mat.NewDense(2, 3, []T{
			11, 12, 13,
			21, 22, 23,
		}),
		grad:         nil,
		requiresGrad: true,
	}
	vs := []*variable[T]{
		{
			value:        mat.NewDense(1, 3, []T{31, 32, 33}),
			grad:         nil,
			requiresGrad: true,
		},
		{
			value:        mat.NewDense(3, 1, []T{41, 42, 43}),
			grad:         nil,
			requiresGrad: true,
		},
	}
	f := NewAppendRows[T](x, vs...)

	assert.Equal(t, []*variable[T]{x, vs[0], vs[1]}, f.Operands())

	y := f.Forward()

	assert.Equal(t, 4, y.Rows())
	assert.Equal(t, 3, y.Columns())
	assert.Equal(t, []T{
		11, 12, 13,
		21, 22, 23,
		31, 32, 33,
		41, 42, 43,
	}, y.Data())

	f.Backward(mat.NewDense(4, 3, []T{
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 0, 1,
	}))

	assert.Equal(t, []T{
		0, 1, 2,
		3, 4, 5,
	}, x.grad.Data())
	assert.Equal(t, []T{6, 7, 8}, vs[0].grad.Data())
	assert.Equal(t, []T{9, 0, 1}, vs[1].grad.Data())
}
