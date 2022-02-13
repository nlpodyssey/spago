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
