// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDensePool(t *testing.T) {
	t.Run("float32", func(t *testing.T) {
		dp := densePool[float32]()
		require.Same(t, densePoolFloat32, dp)
	})

	t.Run("float64", func(t *testing.T) {
		dp := densePool[float64]()
		require.Same(t, densePoolFloat64, dp)
	})
}

func TestDensePool_GetDense(t *testing.T) {
	t.Run("float32", testDensePoolGetDense[float32])
	t.Run("float64", testDensePoolGetDense[float64])
}

func testDensePoolGetDense[T float.DType](t *testing.T) {
	densePool := densePool[T]()

	t.Run("negative rows", func(t *testing.T) {
		require.Panics(t, func() {
			densePool.Get(-1, 1)
		})
	})

	t.Run("negative cols", func(t *testing.T) {
		require.Panics(t, func() {
			densePool.Get(1, -1)
		})
	})

	testCases := []struct {
		rows        int
		cols        int
		expectedCap int
	}{
		{0, 0, 0},
		{0, 1, 0},
		{1, 0, 0},

		{1, 1, 1},

		{1, 2, 3},
		{2, 1, 3},

		{1, 3, 3},
		{3, 1, 3},

		{1, 4, 7},
		{4, 1, 7},

		{5, 5, 31},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.rows, tc.cols), func(t *testing.T) {
			d := densePool.Get(tc.rows, tc.cols)
			assertDenseFromPoolDims(t, tc.rows, tc.cols, tc.expectedCap, d)
		})
	}
}

func assertDenseFromPoolDims[T float.DType](t *testing.T, expectedRows, expectedCols, expectedCap int, d *Dense[T]) {
	t.Helper()
	expectedSize := expectedRows * expectedCols
	assert.Equal(t, expectedRows, d.rows)
	assert.Equal(t, expectedCols, d.cols)
	assert.Len(t, d.data, expectedSize)
	assert.Equal(t, expectedCap, cap(d.data))
}
