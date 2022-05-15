// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStopGrad(t *testing.T) {
	t.Run("float32", testStopGrad[float32])
	t.Run("float64", testStopGrad[float64])
}

func testStopGrad[T mat.DType](t *testing.T) {
	dn := &dummyNode{id: 1}

	sg := StopGrad(dn)
	require.IsType(t, &Wrapper{}, sg)
	w := sg.(*Wrapper)

	assert.Same(t, dn, w.Node)
}

func TestWrapper_Gradients(t *testing.T) {
	t.Run("float32", testWrapperGradients[float32])
	t.Run("float64", testWrapperGradients[float64])
}

func testWrapperGradients[T mat.DType](t *testing.T) {
	grad := mat.NewScalar[T](42)
	dn := &dummyNode{
		grad:         grad,
		requiresGrad: true,
	}
	w := StopGrad(dn)

	assert.False(t, w.RequiresGrad())
	require.Nil(t, w.Grad())
	assert.False(t, w.HasGrad())

	w.AccGrad(mat.NewScalar[T](5))
	require.Nil(t, w.Grad())
	assert.False(t, w.HasGrad())

	w.ZeroGrad()
	require.Nil(t, w.Grad())
	assert.False(t, w.HasGrad())

	// The original node must not be modified
	assert.True(t, dn.RequiresGrad())
	assert.True(t, dn.HasGrad())
	assert.Same(t, grad, dn.Grad())
}
