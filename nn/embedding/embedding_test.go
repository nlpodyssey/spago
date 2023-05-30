// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embedding_test

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn/embedding"
	"github.com/stretchr/testify/assert"
)

func TestEmbedding_Grad(t *testing.T) {
	type T = float32

	m := embedding.New[T](2, 3)

	e1, _ := m.Embedding(0)
	e2, _ := m.Embedding(0)

	assert.False(t, e1.HasGrad())
	assert.False(t, e2.HasGrad())

	assert.Nil(t, e1.Grad())
	assert.Nil(t, e2.Grad())

	e1.ZeroGrad() // At this point, has no effect

	assert.False(t, e1.HasGrad())
	assert.False(t, e2.HasGrad())

	assert.Nil(t, e1.Grad())
	assert.Nil(t, e2.Grad())

	e1.AccGrad(mat.NewDense[T](mat.WithBacking([]T{1, 2, 3})))

	assert.True(t, e1.HasGrad())
	assert.True(t, e2.HasGrad())

	assert.NotNil(t, e1.Grad())
	assert.NotNil(t, e2.Grad())
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{1, 2, 3})), e1.Grad())
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{1, 2, 3})), e2.Grad())
	assert.Same(t, e1.Grad(), e2.Grad())

	e1.AccGrad(mat.NewDense[T](mat.WithBacking([]T{10, 20, 30})))

	assert.True(t, e1.HasGrad())
	assert.True(t, e2.HasGrad())

	assert.NotNil(t, e1.Grad())
	assert.NotNil(t, e2.Grad())
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{11, 22, 33})), e1.Grad())
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{11, 22, 33})), e2.Grad())
	assert.Same(t, e1.Grad(), e2.Grad())

	e1.ZeroGrad()

	assert.False(t, e1.HasGrad())
	assert.False(t, e2.HasGrad())

	assert.Nil(t, e1.Grad())
	assert.Nil(t, e2.Grad())
}
