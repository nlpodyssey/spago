// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestUtils(t *testing.T) {
	t.Run("float32", testUtils[float32])
	t.Run("float64", testUtils[float64])
}

func testUtils[T mat.DType](t *testing.T) {
	t.Run("test `Map2`", func(t *testing.T) {
		ys := Map2(Add,
			[]Node{newScalar[T](1), newScalar[T](2), newScalar[T](3)},
			[]Node{newScalar[T](4), newScalar[T](5), newScalar[T](6)},
		)
		assert.Equal(t, 3, len(ys))
		assert.Equal(t, mat.Float(T(5)), ys[0].Value().Scalar())
		assert.Equal(t, mat.Float(T(7)), ys[1].Value().Scalar())
		assert.Equal(t, mat.Float(T(9)), ys[2].Value().Scalar())
	})

	t.Run("test `Pad`", func(t *testing.T) {
		newEl := func(_ int) Node {
			return newScalar[T](0)
		}
		ys := Pad([]Node{newScalar[T](1), newScalar[T](2), newScalar[T](3)}, 5, newEl)
		assert.Equal(t, 5, len(ys))
		assert.Equal(t, mat.Float(T(1)), ys[0].Value().Scalar())
		assert.Equal(t, mat.Float(T(2)), ys[1].Value().Scalar())
		assert.Equal(t, mat.Float(T(3)), ys[2].Value().Scalar())
		assert.Equal(t, mat.Float(T(0)), ys[3].Value().Scalar())
		assert.Equal(t, mat.Float(T(0)), ys[4].Value().Scalar())
	})

	t.Run("test `Pad` with no need to pad", func(t *testing.T) {
		newEl := func(_ int) Node {
			return newScalar[T](0)
		}
		ys := Pad([]Node{newScalar[T](1), newScalar[T](2), newScalar[T](3)}, 3, newEl)
		assert.Equal(t, 3, len(ys))
		assert.Equal(t, mat.Float(T(1)), ys[0].Value().Scalar())
		assert.Equal(t, mat.Float(T(2)), ys[1].Value().Scalar())
		assert.Equal(t, mat.Float(T(3)), ys[2].Value().Scalar())
	})
}

func TestRowViews(t *testing.T) {
	t.Run("float32", testRowViews[float32])
	t.Run("float64", testRowViews[float64])
}

func testRowViews[T mat.DType](t *testing.T) {
	testCases := []struct {
		x  *mat.Dense[T]
		ys [][]T
	}{
		{mat.NewEmptyDense[T](0, 0), [][]T{}},
		{mat.NewEmptyDense[T](0, 1), [][]T{}},
		{mat.NewEmptyDense[T](1, 0), [][]T{{}}},
		{mat.NewDense[T](1, 1, []T{1}), [][]T{{1}}},
		{mat.NewDense[T](1, 2, []T{1, 2}), [][]T{{1, 2}}},
		{mat.NewDense[T](2, 1, []T{1, 2}), [][]T{{1}, {2}}},
		{
			mat.NewDense[T](2, 2, []T{
				1, 2,
				3, 4,
			}),
			[][]T{
				{1, 2},
				{3, 4},
			},
		},
		{
			mat.NewDense[T](3, 4, []T{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 0, 1, 2,
			}),
			[][]T{
				{1, 2, 3, 4},
				{5, 6, 7, 8},
				{9, 0, 1, 2},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.x.Rows(), tc.x.Columns()), func(t *testing.T) {
			x := NewVariable(tc.x, true)
			ys := RowViews(x)
			assert.Len(t, ys, len(tc.ys))
			for i, yn := range ys {
				y := yn.Value()
				expected := tc.ys[i]

				assert.Equal(t, 1, y.Rows())
				assert.Equal(t, len(expected), y.Columns())
				assert.Equal(t, expected, mat.Data[T](y))
			}
		})
	}
}

func TestColViews(t *testing.T) {
	t.Run("float32", testColViews[float32])
	t.Run("float64", testColViews[float64])
}

func testColViews[T mat.DType](t *testing.T) {
	testCases := []struct {
		x  *mat.Dense[T]
		ys [][]T
	}{
		{mat.NewEmptyDense[T](0, 0), [][]T{}},
		{mat.NewEmptyDense[T](0, 1), [][]T{{}}},
		{mat.NewEmptyDense[T](1, 0), [][]T{}},
		{mat.NewDense[T](1, 1, []T{1}), [][]T{{1}}},
		{mat.NewDense[T](1, 2, []T{1, 2}), [][]T{{1}, {2}}},
		{mat.NewDense[T](2, 1, []T{1, 2}), [][]T{{1, 2}}},
		{
			mat.NewDense[T](2, 2, []T{
				1, 2,
				3, 4,
			}),
			[][]T{
				{1, 3},
				{2, 4},
			},
		},
		{
			mat.NewDense[T](3, 4, []T{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 0, 1, 2,
			}),
			[][]T{
				{1, 5, 9},
				{2, 6, 0},
				{3, 7, 1},
				{4, 8, 2},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.x.Rows(), tc.x.Columns()), func(t *testing.T) {
			x := NewVariable(tc.x, true)
			ys := ColViews(x)
			assert.Len(t, ys, len(tc.ys))
			for i, yn := range ys {
				y := yn.Value()
				expected := tc.ys[i]

				assert.Equal(t, len(expected), y.Rows())
				assert.Equal(t, 1, y.Columns())
				assert.Equal(t, expected, mat.Data[T](y))
			}
		})
	}
}

func newScalar[T mat.DType](v T) Node {
	return NewScalar(mat.NewScalar(v))
}
