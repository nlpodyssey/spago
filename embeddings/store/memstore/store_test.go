// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package memstore_test

import (
	"testing"

	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/embeddings/store/memstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ store.Store = &memstore.Store{}

func TestStore_Name(t *testing.T) {
	s := makeStore(t)
	assert.Equal(t, "test-store", s.Name())
}

func TestStore_DropAll(t *testing.T) {
	s := makeStore(t)

	require.NoError(t, s.Put([]byte{11}, 1))
	require.NoError(t, s.Put([]byte{22}, 2))

	// Data is available before deletion

	var v int

	exists, err := s.Get([]byte{11}, &v)
	require.NoError(t, err)
	assert.True(t, exists)
	assert.Equal(t, 1, v)

	exists, err = s.Get([]byte{22}, &v)
	require.NoError(t, err)
	assert.True(t, exists)
	assert.Equal(t, 2, v)

	// Drop all and check data is not there anymore

	require.NoError(t, s.DropAll())

	exists, err = s.Get([]byte{11}, &v)
	require.NoError(t, err)
	assert.False(t, exists)

	exists, err = s.Get([]byte{22}, &v)
	require.NoError(t, err)
	assert.False(t, exists)
}

func TestStore_Keys(t *testing.T) {
	s := makeStore(t)

	keys, err := s.Keys()
	require.NoError(t, err)
	assert.Nil(t, keys)

	require.NoError(t, s.Put([]byte{1}, 11))

	keys, err = s.Keys()
	require.NoError(t, err)
	assert.Equal(t, [][]byte{{1}}, keys)

	require.NoError(t, s.Put([]byte{2, 3}, 22))

	keys, err = s.Keys()
	require.NoError(t, err)
	assert.Len(t, keys, 2)
	assert.Contains(t, keys, []byte{1})
	assert.Contains(t, keys, []byte{2, 3})
}

func TestStore_KeysCount(t *testing.T) {
	s := makeStore(t)

	n, err := s.KeysCount()
	require.NoError(t, err)
	assert.Equal(t, 0, n)

	require.NoError(t, s.Put([]byte{1}, 11))

	n, err = s.KeysCount()
	require.NoError(t, err)
	assert.Equal(t, 1, n)

	require.NoError(t, s.Put([]byte{2}, 22))

	n, err = s.KeysCount()
	require.NoError(t, err)
	assert.Equal(t, 2, n)
}

func TestStore_PutAndGet(t *testing.T) {
	t.Run("simple int values", func(t *testing.T) {
		s := makeStore(t)
		var v int

		exists, err := s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.False(t, exists)

		require.NoError(t, s.Put([]byte{1, 2}, 11))

		exists, err = s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, 11, v)

		// Overwrite the same key
		require.NoError(t, s.Put([]byte{1, 2}, 22))

		exists, err = s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, 22, v)

		require.NoError(t, s.Put([]byte{3, 4}, 33))

		exists, err = s.Get([]byte{3, 4}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, 33, v)
	})

	t.Run("putting a slice value", func(t *testing.T) {
		s := makeStore(t)

		require.NoError(t, s.Put([]byte{1, 2}, []int{11, 22, 33}))

		var v []int
		exists, err := s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, []int{11, 22, 33}, v)
	})

	t.Run("putting a pointer to slice value", func(t *testing.T) {
		s := makeStore(t)

		require.NoError(t, s.Put([]byte{1, 2}, &[]int{11, 22, 33}))

		var v []int
		exists, err := s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, []int{11, 22, 33}, v)
	})

	t.Run("putting a struct value", func(t *testing.T) {
		s := makeStore(t)

		type st struct{ i int }

		require.NoError(t, s.Put([]byte{1, 2}, st{i: 42}))

		var v st
		exists, err := s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, st{i: 42}, v)
	})

	t.Run("putting a pointer to struct value", func(t *testing.T) {
		s := makeStore(t)

		type st struct{ i int }

		require.NoError(t, s.Put([]byte{1, 2}, &st{i: 42}))

		var v st
		exists, err := s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, st{i: 42}, v)
	})
}

func makeStore(t *testing.T) store.Store {
	t.Helper()

	repo := memstore.NewRepository()

	s, err := repo.Store("test-store")
	require.NoError(t, err)
	return s
}
