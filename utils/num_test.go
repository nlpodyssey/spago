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
