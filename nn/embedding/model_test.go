// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embedding_test

import (
	"bytes"
	"encoding/gob"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/embedding"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ nn.Model = &embedding.Model{}

func TestModel_Count(t *testing.T) {
	type T = float32
	m := embedding.New[T](2, 1)
	assert.Equal(t, 2, m.Size)
}

func TestModel_ClearEmbeddingsWithGrad(t *testing.T) {
	type T = float32
	m := embedding.New[T](1, 3)

	e, _ := m.Embedding(0)
	e.AccGrad(mat.NewVecDense([]T{1, 2, 3}))

	assert.NotNil(t, e.Grad())
	mat.AssertMatrixEquals(t, mat.NewVecDense([]T{1, 2, 3}), e.Grad())

	m.ZeroGrad()

	assert.Nil(t, e.Grad())
}

func TestModel(t *testing.T) {
	t.Run("gob encoding and decoding", func(t *testing.T) {
		type T = float32

		m := embedding.New[T](1, 3)

		e, _ := m.Embedding(0)
		e.ReplaceValue(mat.NewVecDense([]T{1, 2, 3}))
		e.AccGrad(mat.NewVecDense([]T{10, 20, 30}))
		e.SetState([]mat.Matrix{
			mat.Scalar[T](11),
			mat.Scalar[T](22),
		})

		require.Equal(t, m.CountEmbedWithGrad(), 1)

		var buf bytes.Buffer
		require.NoError(t, gob.NewEncoder(&buf).Encode(m))

		var decoded *embedding.Model
		require.NoError(t, gob.NewDecoder(&buf).Decode(&decoded))

		require.NotNil(t, decoded)
		assert.Equal(t, 3, decoded.Dim)
		assert.Equal(t, 1, decoded.Size)

		require.Equal(t, decoded.CountEmbedWithGrad(), 0)
	})
}

func TestModel_TraverseParams(t *testing.T) {
	t.Run("traverse params", func(t *testing.T) {
		type T = float32

		m := embedding.New[T](1, 3)

		e, _ := m.Embedding(0)
		e.AccGrad(mat.NewVecDense([]T{10, 20, 30}))
		e.SetState([]mat.Matrix{
			mat.Scalar[T](11),
			mat.Scalar[T](22),
		})

		require.Equal(t, m.CountEmbedWithGrad(), 1)

		embeddingsWithGrad := make([]*nn.Param, 0)
		nn.ForEachParam(m, func(p *nn.Param) {
			embeddingsWithGrad = append(embeddingsWithGrad, p)
		})
		require.Len(t, embeddingsWithGrad, 1)

		e.ZeroGrad()
		require.Equal(t, m.CountEmbedWithGrad(), 0)

		embeddingsWithGrad = make([]*nn.Param, 0)
		nn.ForEachParam(m, func(p *nn.Param) {
			embeddingsWithGrad = append(embeddingsWithGrad, p)
		})
		require.Len(t, embeddingsWithGrad, 0)
	})
}
