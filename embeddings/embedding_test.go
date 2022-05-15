// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings_test

import (
	"testing"

	"github.com/nlpodyssey/spago/embeddings"
	"github.com/nlpodyssey/spago/embeddings/store/memstore"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/mattest"
	"github.com/nlpodyssey/spago/nn"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ nn.Param = &embeddings.Embedding[string]{}

func TestEmbedding_Value(t *testing.T) {
	type T = float32

	repo := memstore.NewRepository()

	conf := embeddings.Config{
		Size:      3,
		StoreName: "test-store",
		Trainable: true,
	}
	m := embeddings.New[T, string](conf, repo)

	e1, _ := m.Embedding("e")
	e2, _ := m.Embedding("e")

	assert.Nil(t, e1.Value())
	assert.Nil(t, e2.Value())

	assert.Panics(t, func() {
		e1.ApplyDelta(mat.NewVecDense([]T{1, 2, 3}))
	}, "cannot apply delta to embedding not in store")

	// Set a value for the first time

	e1.ReplaceValue(mat.NewVecDense([]T{10, 20, 30}))

	mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{10, 20, 30}), e1.Value())
	mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{10, 20, 30}), e2.Value())

	// Apply delta
	e1.ApplyDelta(mat.NewVecDense([]T{1, 2, 3}))

	mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{9, 18, 27}), e1.Value())
	mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{9, 18, 27}), e2.Value())

	// Set value to nil (weird corner case, but possible)

	e1.ReplaceValue(nil)

	assert.Nil(t, e1.Value())
	assert.Nil(t, e2.Value())

	assert.Panics(t, func() {
		e1.ApplyDelta(mat.NewVecDense([]T{1, 2, 3}))
	}, "cannot apply delta to embedding with nil value")
}

func TestEmbedding_ReplaceValue(t *testing.T) {
	// it must clear grad and payload

	type T = float32

	repo := memstore.NewRepository()

	conf := embeddings.Config{
		Size:      3,
		StoreName: "test-store",
		Trainable: true,
	}
	m := embeddings.New[T, string](conf, repo)

	e, _ := m.Embedding("e")

	payload := &nn.Payload{
		Label: 123,
		Data: []mat.Matrix{
			mat.NewVecDense([]T{11, 22, 33}),
		},
	}

	e.ReplaceValue(mat.NewVecDense([]T{1, 2, 3}))
	e.SetPayload(payload)
	e.AccGrad(mat.NewVecDense([]T{10, 20, 30}))

	mattest.RequireMatrixEquals(t, mat.NewVecDense([]T{1, 2, 3}), e.Value())
	require.True(t, e.HasGrad())
	mattest.RequireMatrixEquals(t, mat.NewVecDense([]T{10, 20, 30}), e.Grad())
	assertPayloadEqual(t, payload, e.Payload())

	e.ReplaceValue(mat.NewVecDense([]T{7, 8, 9}))

	mattest.RequireMatrixEquals(t, mat.NewVecDense([]T{7, 8, 9}), e.Value())
	require.False(t, e.HasGrad())
	require.Nil(t, e.Grad())
	assert.Nil(t, e.Payload())
}

func TestEmbedding_ScalarValue(t *testing.T) {
	type T = float32

	repo := memstore.NewRepository()

	conf := embeddings.Config{
		Size:      1,
		StoreName: "test-store",
		Trainable: true,
	}
	m := embeddings.New[T, string](conf, repo)

	e, _ := m.Embedding("e")

	// Set a value for the first time
	e.ReplaceValue(mat.NewScalar[T](42))
	assert.Equal(t, mat.Float(T(42)), e.Value().Scalar())
}

func TestEmbedding_Grad(t *testing.T) {
	type T = float32

	repo := memstore.NewRepository()

	conf := embeddings.Config{
		Size:      3,
		StoreName: "test-store",
		Trainable: true,
	}
	m := embeddings.New[T, string](conf, repo)

	e1, _ := m.Embedding("e")
	e2, _ := m.Embedding("e")

	assert.False(t, e1.HasGrad())
	assert.False(t, e2.HasGrad())

	assert.Nil(t, e1.Grad())
	assert.Nil(t, e2.Grad())

	e1.ZeroGrad() // At this point, has no effect

	assert.False(t, e1.HasGrad())
	assert.False(t, e2.HasGrad())

	assert.Nil(t, e1.Grad())
	assert.Nil(t, e2.Grad())

	e1.AccGrad(mat.NewVecDense([]T{1, 2, 3}))

	assert.True(t, e1.HasGrad())
	assert.True(t, e2.HasGrad())

	assert.NotNil(t, e1.Grad())
	assert.NotNil(t, e2.Grad())
	mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{1, 2, 3}), e1.Grad())
	mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{1, 2, 3}), e2.Grad())
	assert.Same(t, e1.Grad(), e2.Grad())

	e1.AccGrad(mat.NewVecDense([]T{10, 20, 30}))

	assert.True(t, e1.HasGrad())
	assert.True(t, e2.HasGrad())

	assert.NotNil(t, e1.Grad())
	assert.NotNil(t, e2.Grad())
	mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{11, 22, 33}), e1.Grad())
	mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{11, 22, 33}), e2.Grad())
	assert.Same(t, e1.Grad(), e2.Grad())

	e1.ZeroGrad()

	assert.False(t, e1.HasGrad())
	assert.False(t, e2.HasGrad())

	assert.Nil(t, e1.Grad())
	assert.Nil(t, e2.Grad())
}

func TestEmbedding_RequiresGrad(t *testing.T) {
	t.Run("with Trainable model", func(t *testing.T) {
		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:      3,
			StoreName: "test-store",
			Trainable: true,
		}
		m := embeddings.New[float32, string](conf, repo)

		e, _ := m.Embedding("e")
		assert.True(t, e.RequiresGrad())
	})

	t.Run("with non-Trainable model", func(t *testing.T) {
		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:      3,
			StoreName: "test-store",
			Trainable: false,
		}
		m := embeddings.New[float32, string](conf, repo)

		e, _ := m.Embedding("e")
		assert.False(t, e.RequiresGrad())
	})
}

func TestEmbedding_Name(t *testing.T) {
	t.Run("with string keys", func(t *testing.T) {
		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:      3,
			StoreName: "test-store",
		}
		m := embeddings.New[float32, string](conf, repo)

		foo, _ := m.Embedding("Foo")
		assert.Equal(t, "Foo", foo.Name())
	})

	t.Run("with []byte keys", func(t *testing.T) {
		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:      3,
			StoreName: "test-store",
		}
		m := embeddings.New[float32, []byte](conf, repo)

		foo, _ := m.Embedding([]byte{0xca, 0xfe})
		assert.Equal(t, "CAFE", foo.Name())
	})

	t.Run("with int keys", func(t *testing.T) {
		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:      3,
			StoreName: "test-store",
		}
		m := embeddings.New[float32, int](conf, repo)

		foo, _ := m.Embedding(42)
		assert.Equal(t, "42", foo.Name())
	})
}

func TestEmbedding_Type(t *testing.T) {
	repo := memstore.NewRepository()
	conf := embeddings.Config{
		Size:      3,
		StoreName: "test-store",
	}
	m := embeddings.New[float32, string](conf, repo)

	e, _ := m.Embedding("e")
	assert.Equal(t, nn.Weights, e.Type())
}

func TestEmbedding_SetRequiresGrad(t *testing.T) {
	repo := memstore.NewRepository()
	conf := embeddings.Config{
		Size:      3,
		StoreName: "test-store",
	}
	m := embeddings.New[float32, string](conf, repo)

	e, _ := m.Embedding("e")
	assert.Panics(t, func() {
		e.SetRequiresGrad(true)
	})
	assert.Panics(t, func() {
		e.SetRequiresGrad(false)
	})
}

func TestEmbedding_Payload(t *testing.T) {
	type T = float32

	repo := memstore.NewRepository()

	conf := embeddings.Config{
		Size:      3,
		StoreName: "test-store",
		Trainable: true,
	}
	m := embeddings.New[T, string](conf, repo)

	e1, _ := m.Embedding("e")
	e2, _ := m.Embedding("e")

	assert.Nil(t, e1.Payload())
	assert.Nil(t, e2.Payload())

	// Set a payload for the first time

	payload := &nn.Payload{
		Label: 123,
		Data: []mat.Matrix{
			mat.NewVecDense([]T{1, 2, 3}),
			mat.NewVecDense([]T{4, 5, 6}),
		},
	}
	e1.SetPayload(payload)

	assertPayloadEqual(t, payload, e1.Payload())
	assertPayloadEqual(t, payload, e2.Payload())

	// Clear payload

	e1.ClearPayload()

	assert.Nil(t, e1.Payload())
	assert.Nil(t, e2.Payload())
}

func assertPayloadEqual(t *testing.T, expected, actual *nn.Payload) {
	t.Helper()

	assert.NotNil(t, actual)
	if actual == nil {
		return
	}

	assert.Equal(t, expected.Label, actual.Label)
	assert.Len(t, actual.Data, len(expected.Data))
	if len(actual.Data) != len(expected.Data) {
		return
	}
	for i := range expected.Data {
		assert.Equal(t, expected.Data[i].Data(), actual.Data[i].Data())
	}
}
