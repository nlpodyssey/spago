// Copyright 2019 spaGO Authors. All rights reserved.
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

func TestNewDenseGrad(t *testing.T) {
	t.Run("float32", testNewDenseGrad[float32])
	t.Run("float64", testNewDenseGrad[float64])
}

func testNewDenseGrad[T float.DType](t *testing.T) {
	testCases := []struct {
		value        Matrix
		requiresGrad bool
	}{
		{NewScalar[T](42, WithGrad(true)), true},
		{NewScalar[T](42, WithGrad(false)), false},
	}

	for _, tc := range testCases {
		name := fmt.Sprintf("Var(%g, %v)", tc.value, tc.requiresGrad)
		t.Run(name, func(t *testing.T) {
			v := tc.value
			require.NotNil(t, v)
			assert.Same(t, tc.value, v.Value())
			assert.Nil(t, v.Grad())
			assert.False(t, v.HasGrad())
			assert.Equal(t, tc.requiresGrad, v.RequiresGrad())
		})
	}
}

func TestNewScalarGrad(t *testing.T) {
	t.Run("float32", testNewScalarGrad[float32])
	t.Run("float64", testNewScalarGrad[float64])
}

func testNewScalarGrad[T float.DType](t *testing.T) {
	v := NewScalar(T(42))
	require.NotNil(t, v)
	AssertMatrixEquals(t, NewScalar[T](42), v.Value())
	assert.Nil(t, v.Grad())
	assert.False(t, v.HasGrad())
	assert.False(t, v.RequiresGrad())
}

func TestDense_Gradients(t *testing.T) {
	t.Run("float32", testDenseGradients[float32])
	t.Run("float64", testDenseGradients[float64])
}

func testDenseGradients[T float.DType](t *testing.T) {
	t.Run("with requires gradient true", func(t *testing.T) {
		v := NewScalar[T](42, WithGrad(true))
		require.Nil(t, v.Grad())
		assert.False(t, v.HasGrad())

		v.AccGrad(NewScalar[T](5))
		RequireMatrixEquals(t, NewScalar[T](5), v.Grad())
		assert.True(t, v.HasGrad())

		v.AccGrad(NewScalar[T](10))
		RequireMatrixEquals(t, NewScalar[T](15), v.Grad())
		assert.True(t, v.HasGrad())

		v.ZeroGrad()
		require.Nil(t, v.Grad())
		assert.False(t, v.HasGrad())
	})

	t.Run("with requires gradient false", func(t *testing.T) {
		v := NewScalar[T](42, WithGrad(false))
		require.Nil(t, v.Grad())
		assert.False(t, v.HasGrad())

		v.AccGrad(NewScalar[T](5))
		require.Nil(t, v.Grad())
		assert.False(t, v.HasGrad())

		v.ZeroGrad()
		require.Nil(t, v.Grad())
		assert.False(t, v.HasGrad())
	})
}
