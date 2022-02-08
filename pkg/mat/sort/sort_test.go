// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestDTSlice(t *testing.T) {
	t.Run("float32", testDTSlice[float32])
	t.Run("float64", testDTSlice[float64])
	t.Run("Float", testDTSlice[mat.Float])
}

func testDTSlice[T mat.DType](t *testing.T) {
	s := DTSlice[T]{}
	require.Equal(t, 0, s.Len())

	s = DTSlice[T]{1, 3, 2, mat.NaN[T]()}
	require.Equal(t, 4, s.Len())

	require.True(t, s.Less(0, 1))
	require.False(t, s.Less(1, 0))

	require.False(t, s.Less(1, 2))
	require.True(t, s.Less(2, 1))

	require.False(t, s.Less(0, 3))
	require.True(t, s.Less(3, 0))

	s = DTSlice[T]{1, 3, 2}
	s.Swap(0, 1)
	assert.Equal(t, DTSlice[T]{3, 1, 2}, s)
	s.Swap(1, 2)
	assert.Equal(t, DTSlice[T]{3, 2, 1}, s)

	s.Sort()
	assert.Equal(t, DTSlice[T]{1, 2, 3}, s)
}
