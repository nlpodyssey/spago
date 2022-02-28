// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort_test

import (
	"fmt"
	"github.com/nlpodyssey/spago/utils/sort"
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func TestFloatSlice_Len(t *testing.T) {
	type customFloat32 float32
	type customFloat64 float64

	t.Run("float32", testFloatSliceLen[float32])
	t.Run("float64", testFloatSliceLen[float64])
	t.Run("customFloat32", testFloatSliceLen[customFloat32])
	t.Run("customFloat64", testFloatSliceLen[customFloat64])
}

func testFloatSliceLen[F sort.Float](t *testing.T) {
	testCases := []struct {
		x        []F
		expected int
	}{
		{nil, 0},
		{[]F{}, 0},
		{[]F{1}, 1},
		{[]F{1, 2}, 2},
		{[]F{1, 2, 3}, 3},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%#v", tc.x), func(t *testing.T) {
			x := sort.FloatSlice[F](tc.x)
			actual := x.Len()
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestFloatSlice_Less(t *testing.T) {
	type customFloat32 float32
	type customFloat64 float64

	t.Run("float32", testFloatSliceLess[float32])
	t.Run("float64", testFloatSliceLess[float64])
	t.Run("customFloat32", testFloatSliceLess[customFloat32])
	t.Run("customFloat64", testFloatSliceLess[customFloat64])
}

func testFloatSliceLess[F sort.Float](t *testing.T) {
	nan := F(math.NaN())
	posInf := F(math.Inf(+1))
	negInf := F(math.Inf(-1))

	testCases := []struct {
		x0       F
		x1       F
		expected bool // x0 < x1
	}{
		{0, 0, false},
		{nan, nan, false},
		{posInf, posInf, false},
		{negInf, negInf, false},

		{0, 1, true},
		{1, 0, false},

		{0, nan, false},
		{nan, 0, true},

		{0, posInf, true},
		{posInf, 0, false},

		{0, negInf, false},
		{negInf, 0, true},

		{nan, posInf, true},
		{posInf, nan, false},

		{nan, negInf, true},
		{negInf, nan, false},

		{posInf, negInf, false},
		{negInf, posInf, true},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%g < %g", tc.x0, tc.x1), func(t *testing.T) {
			x := sort.FloatSlice[F]{tc.x0, tc.x1}
			actual := x.Less(0, 1)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestFloatSlice_Swap(t *testing.T) {
	type customFloat32 float32
	type customFloat64 float64

	t.Run("float32", testFloatSliceSwap[float32])
	t.Run("float64", testFloatSliceSwap[float64])
	t.Run("customFloat32", testFloatSliceSwap[customFloat32])
	t.Run("customFloat64", testFloatSliceSwap[customFloat64])
}

func testFloatSliceSwap[F sort.Float](t *testing.T) {
	testCases := []struct {
		x        []F
		i        int
		j        int
		expected []F
	}{
		{[]F{1}, 0, 0, []F{1}},
		{[]F{1, 2}, 0, 0, []F{1, 2}},
		{[]F{1, 2}, 1, 1, []F{1, 2}},

		{[]F{1, 2}, 0, 1, []F{2, 1}},
		{[]F{1, 2}, 1, 0, []F{2, 1}},

		{[]F{1, 2, 3}, 0, 2, []F{3, 2, 1}},
		{[]F{1, 2, 3}, 2, 0, []F{3, 2, 1}},

		{[]F{1, 2, 3}, 1, 2, []F{1, 3, 2}},
		{[]F{1, 2, 3}, 2, 1, []F{1, 3, 2}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%#v swap %d, %d", tc.x, tc.i, tc.j), func(t *testing.T) {
			x := sort.FloatSlice[F](tc.x)
			x.Swap(tc.i, tc.j)
			assert.Equal(t, tc.expected, tc.x)
		})
	}
}

type floatSliceSortTestCase[F sort.Float] struct {
	in        []F
	sorted    []F
	revSorted []F
}

func floatSliceSortTestCases[F sort.Float]() []floatSliceSortTestCase[F] {
	nan := F(math.NaN())
	posInf := F(math.Inf(+1))
	negInf := F(math.Inf(-1))

	return []floatSliceSortTestCase[F]{
		{nil, nil, nil},
		{[]F{}, []F{}, []F{}},
		{[]F{1}, []F{1}, []F{1}},
		{[]F{1, 2, 3}, []F{1, 2, 3}, []F{3, 2, 1}},
		{[]F{2, 1, 3, -1}, []F{-1, 1, 2, 3}, []F{3, 2, 1, -1}},
		{
			[]F{2, nan, 1, posInf, -1, negInf},
			[]F{nan, negInf, -1, 1, 2, posInf},
			[]F{posInf, 2, 1, -1, negInf, nan},
		},
	}
}

func TestFloatSlice_Sort(t *testing.T) {
	type customFloat32 float32
	type customFloat64 float64

	t.Run("float32", testFloatSliceSort[float32])
	t.Run("float64", testFloatSliceSort[float64])
	t.Run("customFloat32", testFloatSliceSort[customFloat32])
	t.Run("customFloat64", testFloatSliceSort[customFloat64])
}

func testFloatSliceSort[F sort.Float](t *testing.T) {
	for _, tc := range floatSliceSortTestCases[F]() {
		t.Run(fmt.Sprintf("%#v", tc.in), func(t *testing.T) {
			x := sort.FloatSlice[F](tc.in)
			x.Sort()
			assertFloatSliceEquals(t, tc.sorted, x)
		})
	}
}

func TestFloatSlice_ReverseSort(t *testing.T) {
	type customFloat32 float32
	type customFloat64 float64

	t.Run("float32", testFloatSliceReverseSort[float32])
	t.Run("float64", testFloatSliceReverseSort[float64])
	t.Run("customFloat32", testFloatSliceReverseSort[customFloat32])
	t.Run("customFloat64", testFloatSliceReverseSort[customFloat64])
}

func testFloatSliceReverseSort[F sort.Float](t *testing.T) {
	for _, tc := range floatSliceSortTestCases[F]() {
		t.Run(fmt.Sprintf("%#v", tc.in), func(t *testing.T) {
			x := sort.FloatSlice[F](tc.in)
			x.ReverseSort()
			assertFloatSliceEquals(t, tc.revSorted, x)
		})
	}
}

func TestSort(t *testing.T) {
	type customFloat32 float32
	type customFloat64 float64

	t.Run("float32", testSort[float32])
	t.Run("float64", testSort[float64])
	t.Run("customFloat32", testSort[customFloat32])
	t.Run("customFloat64", testSort[customFloat64])
}

func testSort[F sort.Float](t *testing.T) {
	for _, tc := range floatSliceSortTestCases[F]() {
		t.Run(fmt.Sprintf("%#v", tc.in), func(t *testing.T) {
			sort.Sort(tc.in)
			assertFloatSliceEquals(t, tc.sorted, tc.in)
		})
	}
}

func TestReverseSort(t *testing.T) {
	type customFloat32 float32
	type customFloat64 float64

	t.Run("float32", testReverseSort[float32])
	t.Run("float64", testReverseSort[float64])
	t.Run("customFloat32", testReverseSort[customFloat32])
	t.Run("customFloat64", testReverseSort[customFloat64])
}

func testReverseSort[F sort.Float](t *testing.T) {
	for _, tc := range floatSliceSortTestCases[F]() {
		t.Run(fmt.Sprintf("%#v", tc.in), func(t *testing.T) {
			sort.ReverseSort(tc.in)
			assertFloatSliceEquals(t, tc.revSorted, tc.in)
		})
	}
}

func assertFloatSliceEquals[F sort.Float](t *testing.T, expected, actual []F) {
	t.Helper()
	for i, a := range actual {
		e := expected[i]
		if math.IsNaN(float64(e)) {
			assert.True(t, math.IsNaN(float64(a)))
		} else {
			assert.Equal(t, e, a)
		}
	}
}
