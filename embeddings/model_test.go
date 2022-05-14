// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings_test

import (
	"bytes"
	"encoding/gob"
	"errors"
	"fmt"
	"github.com/nlpodyssey/spago/mat/mattest"
	"testing"

	"github.com/nlpodyssey/spago/embeddings"
	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/embeddings/store/memstore"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ nn.Model = &embeddings.Model[float32, string]{}

func TestNew(t *testing.T) {
	t.Run("creates a ZeroEmbedding param if UseZeroEmbedding is enabled", func(t *testing.T) {
		type T = float32

		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:             5,
			StoreName:        "test-store",
			UseZeroEmbedding: true,
		}
		m := embeddings.New[T, string](conf, repo)

		require.NotNil(t, m.ZeroEmbedding)
		v := m.ZeroEmbedding.Value()
		require.NotNil(t, v)
		assert.Equal(t, 5, v.Rows())
		assert.Equal(t, 1, v.Columns())
		mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{0, 0, 0, 0, 0}), v)
	})

	t.Run("leaves ZeroEmbedding nil if UseZeroEmbedding is disabled", func(t *testing.T) {
		type T = float32

		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:             5,
			StoreName:        "test-store",
			UseZeroEmbedding: false,
		}
		m := embeddings.New[T, string](conf, repo)

		require.Nil(t, m.ZeroEmbedding)
	})
}

func TestModel_Count(t *testing.T) {
	type T = float32

	repo := memstore.NewRepository()
	conf := embeddings.Config{
		Size:      1,
		StoreName: "test-store",
		Trainable: true,
	}
	m := embeddings.New[T, string](conf, repo)

	assert.Equal(t, 0, m.Count())

	e, _ := m.Embedding("foo")
	e.ReplaceValue(mat.NewScalar[T](11))
	assert.Equal(t, 1, m.Count())

	e, _ = m.Embedding("bar")
	e.ReplaceValue(mat.NewScalar[T](22))
	assert.Equal(t, 2, m.Count())
}

func TestModel_Embedding(t *testing.T) {
	t.Run("setting a Value causes the embedding to be stored", func(t *testing.T) {
		type T = float32

		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:      3,
			StoreName: "test-store",
			Trainable: true,
		}
		m := embeddings.New[T, string](conf, repo)

		e, exists := m.Embedding("e")
		assert.NotNil(t, e)
		assert.False(t, exists)

		// If the embedding is not modified, it should still not exist
		e2, exists := m.Embedding("e")
		assert.NotNil(t, e2)
		assert.False(t, exists)

		// Modify the value
		e.ReplaceValue(mat.NewVecDense([]T{1, 2, 3}))

		// Now it must exist
		e2, exists = m.Embedding("e")
		assert.NotNil(t, e2)
		assert.True(t, exists)
	})

	t.Run("setting a Payload causes the embedding to be stored", func(t *testing.T) {
		type T = float32

		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:      1,
			StoreName: "test-store",
			Trainable: true,
		}
		m := embeddings.New[T, string](conf, repo)

		e, exists := m.Embedding("e")
		assert.NotNil(t, e)
		assert.False(t, exists)

		// If the embedding is not modified, it should still not exist
		e2, exists := m.Embedding("e")
		assert.NotNil(t, e2)
		assert.False(t, exists)

		// Modify the value
		e.SetPayload(&nn.Payload[T]{
			Label: 123,
			Data: []mat.Matrix{
				mat.NewScalar[T](11),
				mat.NewScalar[T](22),
			},
		})

		// Now it must exist
		e2, exists = m.Embedding("e")
		assert.NotNil(t, e2)
		assert.True(t, exists)
	})

	t.Run("embeddings with a gradient are memoized", func(t *testing.T) {
		type T = float32

		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:      3,
			StoreName: "test-store",
			Trainable: true,
		}
		m := embeddings.New[T, string](conf, repo)

		e, _ := m.Embedding("e")

		e2, _ := m.Embedding("e")
		assert.NotSame(t, e, e2, "no grad: not memoized")

		e.AccGrad(mat.NewVecDense([]T{1, 2, 3}))

		e2, _ = m.Embedding("e")
		assert.Same(t, e, e2, "has grad: memoized")

		e.ZeroGrad()

		e2, _ = m.Embedding("e")
		assert.NotSame(t, e, e2, "grad has been zeroed: not memoized")
	})
}

func TestModel_Encode(t *testing.T) {
	t.Run("with UseZeroEmbedding enabled", func(t *testing.T) {
		type T = float32

		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:             3,
			StoreName:        "test-store",
			UseZeroEmbedding: true,
		}
		model := embeddings.New[T, string](conf, repo)

		e, _ := model.Embedding("foo")
		e.ReplaceValue(mat.NewVecDense([]T{1, 2, 3}))

		result := model.Encode([]string{"foo", "bar", "foo"})
		require.Len(t, result, 3)

		assert.NotNil(t, result[0])
		assert.NotNil(t, result[0].Value())
		mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{1, 2, 3}), result[0].Value())

		assert.NotNil(t, result[1])
		assert.NotNil(t, result[1].Value())
		mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{0, 0, 0}), result[1].Value())

		assert.NotNil(t, result[2])
		assert.NotNil(t, result[2].Value())
		mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{1, 2, 3}), result[2].Value())
	})

	t.Run("with UseZeroEmbedding disabled", func(t *testing.T) {
		type T = float32

		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:             3,
			StoreName:        "test-store",
			UseZeroEmbedding: false,
		}
		model := embeddings.New[T, string](conf, repo)

		e, _ := model.Embedding("foo")
		e.ReplaceValue(mat.NewVecDense([]T{1, 2, 3}))

		result := model.Encode([]string{"foo", "bar", "foo"})
		require.Len(t, result, 3)

		assert.NotNil(t, result[0])
		assert.NotNil(t, result[0].Value())
		mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{1, 2, 3}), result[0].Value())

		assert.Nil(t, result[1])

		assert.NotNil(t, result[2])
		assert.NotNil(t, result[2].Value())
		mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{1, 2, 3}), result[2].Value())
	})
}

func TestModel_ClearEmbeddingsWithGrad(t *testing.T) {
	type T = float32

	repo := memstore.NewRepository()
	conf := embeddings.Config{
		Size:      3,
		StoreName: "test-store",
		Trainable: true,
	}
	m := embeddings.New[T, string](conf, repo)

	e, _ := m.Embedding("e")
	e.AccGrad(mat.NewVecDense([]T{1, 2, 3}))

	assert.NotNil(t, e.Grad())
	mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{1, 2, 3}), e.Grad())

	m.ClearEmbeddingsWithGrad()

	assert.Nil(t, e.Grad())
}

func TestModel_UseRepository(t *testing.T) {
	type T = float32

	t.Run("when Store is nil", func(t *testing.T) {
		repo := memstore.NewRepository()
		m := &embeddings.Model[T, string]{
			Config: embeddings.Config{
				StoreName: "foo",
			},
			Store: nil,
		}

		err := m.UseRepository(repo)
		require.NoError(t, err)

		st, err := repo.Store("foo")
		require.NoError(t, err)
		expected := &store.PreventStoreMarshaling{Store: st}
		require.Equal(t, expected, m.Store)
	})

	t.Run("when Store is PreventStoreMarshaling{nil}", func(t *testing.T) {
		repo := memstore.NewRepository()
		m := &embeddings.Model[T, string]{
			Config: embeddings.Config{
				StoreName: "foo",
			},
			Store: store.PreventStoreMarshaling{Store: nil},
		}

		err := m.UseRepository(repo)
		require.NoError(t, err)

		st, err := repo.Store("foo")
		require.NoError(t, err)
		expected := &store.PreventStoreMarshaling{Store: st}
		require.Equal(t, expected, m.Store)
	})

	t.Run("when Store is *PreventStoreMarshaling{nil}", func(t *testing.T) {
		repo := memstore.NewRepository()
		m := &embeddings.Model[T, string]{
			Config: embeddings.Config{
				StoreName: "foo",
			},
			Store: &store.PreventStoreMarshaling{Store: nil},
		}

		err := m.UseRepository(repo)
		require.NoError(t, err)

		st, err := repo.Store("foo")
		require.NoError(t, err)
		expected := &store.PreventStoreMarshaling{Store: st}
		require.Equal(t, expected, m.Store)
	})

	t.Run("when Store is not nil", func(t *testing.T) {
		repo := memstore.NewRepository()

		st, err := repo.Store("foo")
		require.NoError(t, err)

		m := &embeddings.Model[T, string]{
			Config: embeddings.Config{
				StoreName: "foo",
			},
			Store: st,
		}

		err = m.UseRepository(repo)
		require.Error(t, err)
		require.Same(t, st, m.Store, "Store was not modified")
	})

	t.Run("when Store is PreventStoreMarshaling{non-nil}", func(t *testing.T) {
		repo := memstore.NewRepository()

		st, err := repo.Store("foo")
		require.NoError(t, err)

		mst := store.PreventStoreMarshaling{Store: st}

		m := &embeddings.Model[T, string]{
			Config: embeddings.Config{
				StoreName: "foo",
			},
			Store: mst,
		}

		err = m.UseRepository(repo)
		require.Error(t, err)
		require.Equal(t, mst, m.Store, "Store was not modified")
	})

	t.Run("when Store is *PreventStoreMarshaling{nil}", func(t *testing.T) {
		repo := memstore.NewRepository()

		st, err := repo.Store("foo")
		require.NoError(t, err)

		mst := &store.PreventStoreMarshaling{Store: st}

		m := &embeddings.Model[T, string]{
			Config: embeddings.Config{
				StoreName: "foo",
			},
			Store: mst,
		}

		err = m.UseRepository(repo)
		require.Error(t, err)
		require.Same(t, mst, m.Store, "Store was not modified")
	})

	t.Run("error getting Store from Repository", func(t *testing.T) {
		storeError := errors.New("repo store test error")
		repo := repoStub{
			fnStore: func(name string) (store.Store, error) {
				require.Equal(t, "foo", name)
				return nil, storeError
			},
			fnDropAll: func() error {
				msg := "DropAll should not be invoked"
				t.Fatal(msg)
				return fmt.Errorf(msg)
			},
		}

		m := &embeddings.Model[T, string]{
			Config: embeddings.Config{
				StoreName: "foo",
			},
			Store: nil,
		}

		err := m.UseRepository(repo)
		require.ErrorIs(t, err, storeError)
		assert.Nil(t, m.Store)
	})
}

func TestModel(t *testing.T) {
	t.Run("gob encoding and decoding", func(t *testing.T) {
		type T = float32

		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:             3,
			UseZeroEmbedding: true,
			StoreName:        "test-store",
			Trainable:        true,
		}
		m := embeddings.New[T, string](conf, repo)

		e, _ := m.Embedding("e")
		e.ReplaceValue(mat.NewVecDense([]T{1, 2, 3}))
		e.AccGrad(mat.NewVecDense([]T{10, 20, 30}))
		e.SetPayload(&nn.Payload[T]{
			Label: 123,
			Data: []mat.Matrix{
				mat.NewScalar[T](11),
				mat.NewScalar[T](22),
			},
		})

		require.NotNil(t, m.ZeroEmbedding)
		require.NotNil(t, m.Store)
		require.Len(t, m.EmbeddingsWithGrad, 1)

		var buf bytes.Buffer
		require.NoError(t, gob.NewEncoder(&buf).Encode(m))

		var decoded *embeddings.Model[T, string]
		require.NoError(t, gob.NewDecoder(&buf).Decode(&decoded))

		require.NotNil(t, decoded)
		assert.Equal(t, conf, decoded.Config)

		require.NotNil(t, decoded.ZeroEmbedding)
		mattest.AssertMatrixEquals(t, mat.NewVecDense([]T{0, 0, 0}), decoded.ZeroEmbedding.Value())

		require.NotNil(t, decoded.Store)
		assert.Nil(t, decoded.Store.(store.PreventStoreMarshaling).Store)

		require.Nil(t, decoded.EmbeddingsWithGrad)
	})
}

type repoStub struct {
	fnStore   func(name string) (store.Store, error)
	fnDropAll func() error
}

func (r repoStub) Store(name string) (store.Store, error) {
	return r.fnStore(name)
}

func (r repoStub) DropAll() error {
	return r.fnDropAll()
}
