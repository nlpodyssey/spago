// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestIsVector(t *testing.T) {
	t.Run("float32", testIsVector[float32])
	t.Run("float64", testIsVector[float64])
}

func testIsVector[T DType](t *testing.T) {
	testCases := []struct {
		r int
		c int
		b bool
	}{
		{0, 0, false},
		{0, 1, true},
		{1, 0, true},
		{1, 1, true},
		{1, 2, true},
		{2, 1, true},
		{1, 9, true},
		{9, 1, true},
		{2, 2, false},
		{3, 4, false},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.r, tc.c), func(t *testing.T) {
			d := NewEmptyDense[T](tc.r, tc.c)
			require.Equal(t, tc.b, IsVector[T](d))
		})
	}
}

func TestIsScalar(t *testing.T) {
	t.Run("float32", testIsScalar[float32])
	t.Run("float64", testIsScalar[float64])
}

func testIsScalar[T DType](t *testing.T) {
	testCases := []struct {
		r int
		c int
		b bool
	}{
		{0, 0, false},
		{0, 1, false},
		{1, 0, false},
		{1, 1, true},
		{1, 2, false},
		{2, 1, false},
		{2, 2, false},
		{3, 4, false},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.r, tc.c), func(t *testing.T) {
			d := NewEmptyDense[T](tc.r, tc.c)
			require.Equal(t, tc.b, IsScalar[T](d))
		})
	}
}

func TestSameDims(t *testing.T) {
	t.Run("float32", testSameDims[float32])
	t.Run("float64", testSameDims[float64])
}

func testSameDims[T DType](t *testing.T) {
	t.Run("different dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](3, 2)
		assert.False(t, SameDims[T](a, b))
		assert.False(t, SameDims[T](b, a))
	})

	t.Run("same dimensions", func(t *testing.T) {
		a := NewEmptyDense[T](2, 3)
		b := NewEmptyDense[T](2, 3)
		assert.True(t, SameDims[T](a, b))
		assert.True(t, SameDims[T](b, a))
	})
}

func TestVectorsOfSameSize(t *testing.T) {
	t.Run("float32", testVectorsOfSameSize[float32])
	t.Run("float64", testVectorsOfSameSize[float64])
}

func testVectorsOfSameSize[T DType](t *testing.T) {
	testCases := []struct {
		a Matrix[T]
		b Matrix[T]
		y bool
	}{
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 0), false},
		{NewEmptyDense[T](0, 0), NewEmptyVecDense[T](0), false},
		{NewEmptyVecDense[T](0), NewEmptyDense[T](0, 0), false},
		{NewEmptyDense[T](2, 3), NewEmptyDense[T](2, 3), false},
		{NewEmptyDense[T](1, 2), NewEmptyDense[T](1, 3), false},
		{NewEmptyDense[T](1, 3), NewEmptyDense[T](1, 2), false},
		{NewEmptyDense[T](1, 2), NewEmptyDense[T](1, 2), true},
		{NewEmptyDense[T](1, 2), NewEmptyDense[T](2, 1), true},
		{NewEmptyDense[T](2, 1), NewEmptyDense[T](1, 2), true},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d and %d x %d", tc.a.Rows(), tc.a.Columns(), tc.b.Rows(), tc.b.Columns()), func(t *testing.T) {
			y := VectorsOfSameSize(tc.a, tc.b)
			assert.Equal(t, tc.y, y)
		})
	}
}

func TestConcatV(t *testing.T) {
	t.Run("float32", testConcatV[float32])
	t.Run("float64", testConcatV[float64])
}

func testConcatV[T DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		var d Matrix[T] = NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			ConcatV(d)
		})
	})

	testCases := []struct {
		xs []Matrix[T]
		y  []T
	}{
		{[]Matrix[T]{}, []T{}},
		{[]Matrix[T]{NewEmptyDense[T](0, 1)}, []T{}},
		{[]Matrix[T]{NewEmptyDense[T](1, 0)}, []T{}},
		{[]Matrix[T]{NewDense[T](1, 1, []T{1})}, []T{1}},
		{
			[]Matrix[T]{
				NewDense[T](1, 1, []T{1}),
				NewDense[T](1, 1, []T{2}),
			},
			[]T{1, 2},
		},
		{
			[]Matrix[T]{
				NewDense[T](1, 2, []T{1, 2}),
				NewDense[T](2, 1, []T{3, 4}),
			},
			[]T{1, 2, 3, 4},
		},
		{
			[]Matrix[T]{
				NewDense[T](1, 1, []T{1}),
				NewDense[T](2, 1, []T{2, 3}),
				NewDense[T](1, 3, []T{4, 5, 6}),
			},
			[]T{1, 2, 3, 4, 5, 6},
		},
	}

	for _, tc := range testCases {
		name := "["
		for _, x := range tc.xs {
			name += fmt.Sprintf(" (%d x %d)", x.Rows(), x.Columns())
		}
		name += " ]"
		t.Run(name, func(t *testing.T) {
			y := ConcatV[T](tc.xs...)
			assert.Equal(t, len(tc.y), y.Rows())
			assert.Equal(t, 1, y.Columns())
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestStack(t *testing.T) {
	t.Run("float32", testStack[float32])
	t.Run("float64", testStack[float64])
}

func testStack[T DType](t *testing.T) {
	t.Run("non-vector matrix", func(t *testing.T) {
		var d Matrix[T] = NewEmptyDense[T](2, 3)
		require.Panics(t, func() {
			Stack(d)
		})
	})

	t.Run("vectors of different sizes", func(t *testing.T) {
		var a Matrix[T] = NewEmptyDense[T](1, 2)
		var b Matrix[T] = NewEmptyDense[T](1, 3)
		require.Panics(t, func() {
			Stack(a, b)
		})
	})

	testCases := []struct {
		xs []Matrix[T]
		y  []T
	}{
		{[]Matrix[T]{}, []T{}},
		{[]Matrix[T]{NewEmptyDense[T](0, 1)}, []T{}},
		{[]Matrix[T]{NewEmptyDense[T](1, 0)}, []T{}},
		{[]Matrix[T]{NewDense[T](1, 1, []T{1})}, []T{1}},
		{
			[]Matrix[T]{
				NewDense[T](1, 1, []T{1}),
				NewDense[T](1, 1, []T{2}),
			},
			[]T{
				1,
				2,
			},
		},
		{
			[]Matrix[T]{
				NewDense[T](2, 1, []T{1, 2}),
				NewDense[T](2, 1, []T{3, 4}),
			},
			[]T{
				1, 2,
				3, 4,
			},
		},
		{
			[]Matrix[T]{
				NewDense[T](2, 1, []T{1, 2}),
				NewDense[T](1, 2, []T{3, 4}),
			},
			[]T{
				1, 2,
				3, 4,
			},
		},
	}

	for _, tc := range testCases {
		name := "["
		for _, x := range tc.xs {
			name += fmt.Sprintf(" (%d x %d)", x.Rows(), x.Columns())
		}
		name += " ]"
		t.Run(name, func(t *testing.T) {
			y := Stack[T](tc.xs...)
			assert.Equal(t, len(tc.xs), y.Rows())
			cols := 0
			if len(tc.xs) > 0 {
				cols = tc.xs[0].Size()
			}
			assert.Equal(t, cols, y.Columns())
			assert.Equal(t, tc.y, y.Data())
		})
	}
}

func TestEqual(t *testing.T) {
	t.Run("float32", testEqual[float32])
	t.Run("float64", testEqual[float64])
}

func testEqual[T DType](t *testing.T) {
	testCases := []struct {
		a, b     Matrix[T]
		expected bool
	}{
		{NewEmptyDense[T](0, 0), NewEmptyDense[T](0, 0), true},
		{NewEmptyDense[T](0, 1), NewEmptyDense[T](0, 1), true},
		{NewEmptyDense[T](1, 0), NewEmptyDense[T](1, 0), true},
		{NewEmptyDense[T](1, 1), NewEmptyDense[T](1, 2), false},
		{NewEmptyDense[T](1, 1), NewEmptyDense[T](2, 1), false},
		{NewEmptyDense[T](1, 2), NewEmptyDense[T](2, 1), false},
		{NewDense[T](1, 1, []T{42}), NewDense[T](1, 1, []T{42}), true},
		{NewDense[T](1, 1, []T{42}), NewDense[T](1, 1, []T{41}), false},
		{
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			true,
		},
		{
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 6,
			}),
			NewDense[T](2, 3, []T{
				1, 2, 3,
				4, 5, 7,
			}),
			false,
		},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("Equal(%dx%d, %dx%d) == %v",
			tc.a.Rows(), tc.a.Columns(), tc.b.Rows(), tc.b.Columns(), tc.expected)
		t.Run(name, func(t *testing.T) {
			assert.Equal(t, tc.expected, Equal(tc.a, tc.b), "a vs b")
			assert.Equal(t, tc.expected, Equal(tc.b, tc.a), "b vs a")
		})
	}
}
