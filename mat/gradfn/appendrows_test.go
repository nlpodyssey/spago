// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"github.com/stretchr/testify/assert"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

func TestAppendRowsForward(t *testing.T) {
	t.Run("float32", testAppendRowsForward[float32])
	t.Run("float64", testAppendRowsForward[float64])
}

func testAppendRowsForward[T float.DType](t *testing.T) {
	x := mat.Tensor(mat.NewDense[T](mat.WithShape(2, 3), mat.WithBacking([]T{
		11, 12, 13,
		21, 22, 23,
	}), mat.WithGrad(true)))

	vs := []mat.Tensor{
		mat.NewDense[T](mat.WithShape(1, 3), mat.WithBacking([]T{31, 32, 33}), mat.WithGrad(true)),
		mat.NewDense[T](mat.WithShape(3, 1), mat.WithBacking([]T{41, 42, 43}), mat.WithGrad(true)),
	}
	f := NewAppendRows(x, vs...)

	assert.Equal(t, []mat.Tensor{x, vs[0], vs[1]}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)

	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithShape(4, 3), mat.WithBacking([]T{
		11, 12, 13,
		21, 22, 23,
		31, 32, 33,
		41, 42, 43,
	})), y.(mat.Matrix))

	err = f.Backward(mat.NewDense[T](mat.WithShape(4, 3), mat.WithBacking([]T{
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 0, 1,
	})))
	assert.NoError(t, err)

	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithShape(2, 3), mat.WithBacking([]T{
		0, 1, 2,
		3, 4, 5,
	})), x.Grad().(mat.Matrix))
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{6, 7, 8})).T(), vs[0].Grad().(mat.Matrix))
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{9, 0, 1})).T(), vs[1].Grad().(mat.Matrix))
}
