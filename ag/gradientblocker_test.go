// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStopGrad(t *testing.T) {
	t.Run("float32", testStopGrad[float32])
	t.Run("float64", testStopGrad[float64])
}

func testStopGrad[T float.DType](t *testing.T) {
	dn := mat.Scalar(T(42))

	sg := StopGrad(dn)
	require.IsType(t, &GradientBlocker{}, sg)
	w := sg.(*GradientBlocker)

	assert.Same(t, dn, w.Tensor)
}

func TestWrapper_Gradients(t *testing.T) {
	t.Run("float32", testWrapperGradients[float32])
	t.Run("float64", testWrapperGradients[float64])
}

func testWrapperGradients[T float.DType](t *testing.T) {
	value := mat.Scalar[T](24)
	grad := mat.Scalar[T](42)
	value.SetRequiresGrad(true)
	value.AccGrad(grad)
	w := StopGrad(grad)

	assert.False(t, w.RequiresGrad())
	require.Nil(t, w.Grad())
	assert.False(t, w.HasGrad())

	w.AccGrad(mat.Scalar[T](5))
	require.Nil(t, w.Grad())
	assert.False(t, w.HasGrad())

	w.ZeroGrad()
	require.Nil(t, w.Grad())
	assert.False(t, w.HasGrad())

	// The original node must not be modified
	assert.True(t, value.RequiresGrad())
	assert.True(t, value.HasGrad())
	assert.Equal(t, grad, value.Grad())
}
