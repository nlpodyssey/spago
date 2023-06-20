// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestUtils(t *testing.T) {
	t.Run("float32", testUtils[float32])
	t.Run("float64", testUtils[float64])
}

func testUtils[T float.DType](t *testing.T) {
	t.Run("test `Map2`", func(t *testing.T) {
		ys := Map2(Add,
			[]mat.Tensor{newScalar[T](1), newScalar[T](2), newScalar[T](3)},
			[]mat.Tensor{newScalar[T](4), newScalar[T](5), newScalar[T](6)},
		)
		assert.Equal(t, 3, len(ys))
		assert.Equal(t, float.Interface(T(5)), ys[0].Value().Item())
		assert.Equal(t, float.Interface(T(7)), ys[1].Value().Item())
		assert.Equal(t, float.Interface(T(9)), ys[2].Value().Item())
	})

	t.Run("test `Pad`", func(t *testing.T) {
		newEl := func(_ int) mat.Tensor {
			return newScalar[T](0)
		}
		ys := Pad([]mat.Tensor{newScalar[T](1), newScalar[T](2), newScalar[T](3)}, 5, newEl)
		assert.Equal(t, 5, len(ys))
		assert.Equal(t, float.Interface(T(1)), ys[0].Value().Item())
		assert.Equal(t, float.Interface(T(2)), ys[1].Value().Item())
		assert.Equal(t, float.Interface(T(3)), ys[2].Value().Item())
		assert.Equal(t, float.Interface(T(0)), ys[3].Value().Item())
		assert.Equal(t, float.Interface(T(0)), ys[4].Value().Item())
	})

	t.Run("test `Pad` with no need to pad", func(t *testing.T) {
		newEl := func(_ int) mat.Tensor {
			return newScalar[T](0)
		}
		ys := Pad([]mat.Tensor{newScalar[T](1), newScalar[T](2), newScalar[T](3)}, 3, newEl)
		assert.Equal(t, 3, len(ys))
		assert.Equal(t, float.Interface(T(1)), ys[0].Value().Item())
		assert.Equal(t, float.Interface(T(2)), ys[1].Value().Item())
		assert.Equal(t, float.Interface(T(3)), ys[2].Value().Item())
	})
}

func TestRowViews(t *testing.T) {
	t.Run("float32", testRowViews[float32])
	t.Run("float64", testRowViews[float64])
}

func testRowViews[T float.DType](t *testing.T) {
	testCases := []struct {
		x  *mat.Dense[T]
		ys [][]T
	}{
		{mat.NewDense[T](mat.WithShape(0, 0)), [][]T{}},
		{mat.NewDense[T](mat.WithShape(0, 1)), [][]T{}},
		{mat.NewDense[T](mat.WithShape(1, 0)), [][]T{{}}},
		{mat.NewDense[T](mat.WithShape(1, 1), mat.WithBacking([]T{1})), [][]T{{1}}},
		{mat.NewDense[T](mat.WithShape(1, 2), mat.WithBacking([]T{1, 2})), [][]T{{1, 2}}},
		{mat.NewDense[T](mat.WithShape(2, 1), mat.WithBacking([]T{1, 2})), [][]T{{1}, {2}}},
		{
			mat.NewDense[T](mat.WithShape(2, 2), mat.WithBacking([]T{
				1, 2,
				3, 4,
			})),
			[][]T{
				{1, 2},
				{3, 4},
			},
		},
		{
			mat.NewDense[T](mat.WithShape(3, 4), mat.WithBacking([]T{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 0, 1, 2,
			})),
			[][]T{
				{1, 2, 3, 4},
				{5, 6, 7, 8},
				{9, 0, 1, 2},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.x.Shape()[0], tc.x.Shape()[1]), func(t *testing.T) {
			x := tc.x
			x.SetRequiresGrad(true)
			ys := RowViews(x)
			assert.Len(t, ys, len(tc.ys))
			for i, yn := range ys {
				y := yn.Value()
				expected := tc.ys[i]

				assert.Equal(t, 1, y.Shape()[0])
				assert.Equal(t, len(expected), y.Shape()[1])
				assert.Equal(t, expected, mat.Data[T](y))
			}
		})
	}
}

func TestColViews(t *testing.T) {
	t.Run("float32", testColViews[float32])
	t.Run("float64", testColViews[float64])
}

func testColViews[T float.DType](t *testing.T) {
	testCases := []struct {
		x  *mat.Dense[T]
		ys [][]T
	}{
		{mat.NewDense[T](mat.WithShape(0, 0)), [][]T{}},
		{mat.NewDense[T](mat.WithShape(0, 1)), [][]T{{}}},
		{mat.NewDense[T](mat.WithShape(1, 0)), [][]T{}},
		{mat.NewDense[T](mat.WithShape(1, 1), mat.WithBacking([]T{1})), [][]T{{1}}},
		{mat.NewDense[T](mat.WithShape(1, 2), mat.WithBacking([]T{1, 2})), [][]T{{1}, {2}}},
		{mat.NewDense[T](mat.WithShape(2, 1), mat.WithBacking([]T{1, 2})), [][]T{{1, 2}}},
		{
			mat.NewDense[T](mat.WithShape(2, 2), mat.WithBacking([]T{
				1, 2,
				3, 4,
			})),
			[][]T{
				{1, 3},
				{2, 4},
			},
		},
		{
			mat.NewDense[T](mat.WithShape(3, 4), mat.WithBacking([]T{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 0, 1, 2,
			})),
			[][]T{
				{1, 5, 9},
				{2, 6, 0},
				{3, 7, 1},
				{4, 8, 2},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.x.Shape()[0], tc.x.Shape()[1]), func(t *testing.T) {
			x := tc.x
			x.SetRequiresGrad(true)
			ys := ColViews(x)
			assert.Len(t, ys, len(tc.ys))
			for i, yn := range ys {
				y := yn.Value()
				expected := tc.ys[i]

				assert.Equal(t, len(expected), y.Shape()[0])
				assert.Equal(t, 1, y.Shape()[1])
				assert.Equal(t, expected, mat.Data[T](y.(mat.Matrix)))
			}
		})
	}
}

func newScalar[T float.DType](v T) mat.Tensor {
	return mat.Scalar(v)
}
