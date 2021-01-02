// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMinInt(t *testing.T) {
	assert.Equal(t, 1, MinInt(1, 2))
	assert.Equal(t, 1, MinInt(2, 1))
}

func TestSumInt(t *testing.T) {
	assert.Equal(t, 0, SumInt([]int{}))
	assert.Equal(t, 0, SumInt([]int{0, 0}))
	assert.Equal(t, 6, SumInt([]int{1, 2, 3}))
}

func TestReverseIntSlice(t *testing.T) {
	assert.Equal(t, []int{}, ReverseIntSlice([]int{}))
	assert.Equal(t, []int{1}, ReverseIntSlice([]int{1}))
	assert.Equal(t, []int{2, 1}, ReverseIntSlice([]int{1, 2}))

	source := []int{1, 2, 3}
	assert.Equal(t, []int{3, 2, 1}, ReverseIntSlice(source))
	assert.Equal(t, []int{1, 2, 3}, source, "the source is not modified")
}

func TestMakeIndices(t *testing.T) {
	assert.Equal(t, []int{}, MakeIndices(0))
	assert.Equal(t, []int{0}, MakeIndices(1))
	assert.Equal(t, []int{0, 1}, MakeIndices(2))
	assert.Equal(t, []int{0, 1, 2}, MakeIndices(3))
}

func TestMakeIntMatrix(t *testing.T) {
	assert.Equal(t, [][]int{}, MakeIntMatrix(0, 0))
	assert.Equal(t, [][]int{}, MakeIntMatrix(0, 1))
	assert.Equal(t, [][]int{{}}, MakeIntMatrix(1, 0))
	assert.Equal(t, [][]int{{0}}, MakeIntMatrix(1, 1))
	assert.Equal(t, [][]int{{0, 0, 0}, {0, 0, 0}}, MakeIntMatrix(2, 3))
}

func TestContainsInt(t *testing.T) {
	assert.False(t, ContainsInt([]int{}, 0))
	assert.False(t, ContainsInt([]int{}, 1))

	assert.True(t, ContainsInt([]int{1}, 1))
	assert.False(t, ContainsInt([]int{1}, 0))

	assert.True(t, ContainsInt([]int{1, 2}, 1))
	assert.True(t, ContainsInt([]int{1, 2}, 2))
	assert.False(t, ContainsInt([]int{1, 2}, 0))
}

func TestGetNeighborsIndices(t *testing.T) {
	assert.Equal(t, []int{}, GetNeighborsIndices(0, 0, 0))
	assert.Equal(t, []int{3, 4, 0, 1}, GetNeighborsIndices(5, 0, 2))
	assert.Equal(t, []int{2, 3, 4, 0, 1, 2}, GetNeighborsIndices(5, 0, 3))
}

func TestAbs(t *testing.T) {
	assert.Equal(t, 0, Abs(0))
	assert.Equal(t, 42, Abs(42))
	assert.Equal(t, 42, Abs(-42))
}
