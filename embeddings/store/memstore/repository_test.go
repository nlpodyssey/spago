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

var _ store.Repository = &memstore.Repository{}

func TestRepository_Store(t *testing.T) {
	t.Run("it returns memoized instances when called multiple times", func(t *testing.T) {
		repo := memstore.NewRepository()

		s1a, err := repo.Store("s1")
		require.NoError(t, err)

		s1b, err := repo.Store("s1")
		require.NoError(t, err)
		assert.Same(t, s1a, s1b)

		s2, err := repo.Store("s2")
		require.NoError(t, err)
		assert.NotSame(t, s1a, s2)
	})

	t.Run("data in each store is isolated", func(t *testing.T) {
		repo := memstore.NewRepository()

		// Insert in different stores

		a, err := repo.Store("")
		require.NoError(t, err)
		require.NoError(t, a.Put([]byte{11}, 1))

		b, err := repo.Store("b")
		require.NoError(t, err)
		require.NoError(t, b.Put([]byte{22}, 2))
		require.NoError(t, b.Put([]byte{33}, 3))

		c, err := repo.Store("c")
		require.NoError(t, err)
		require.NoError(t, c.Put([]byte{44}, 4))

		// Read from the stores

		var v int

		keys, err := a.Keys()
		require.NoError(t, err)
		assert.Equal(t, [][]byte{{11}}, keys)

		exists, err := a.Get([]byte{11}, &v)
		require.NoError(t, err)
		assert.True(t, exists)
		assert.Equal(t, 1, v)

		keys, err = b.Keys()
		require.NoError(t, err)
		assert.Len(t, keys, 2)
		assert.Contains(t, keys, []byte{22})
		assert.Contains(t, keys, []byte{33})

		exists, err = b.Get([]byte{22}, &v)
		require.NoError(t, err)
		assert.True(t, exists)
		assert.Equal(t, 2, v)

		exists, err = b.Get([]byte{33}, &v)
		require.NoError(t, err)
		assert.True(t, exists)
		assert.Equal(t, 3, v)

		keys, err = c.Keys()
		require.NoError(t, err)
		assert.Equal(t, [][]byte{{44}}, keys)

		exists, err = c.Get([]byte{44}, &v)
		require.NoError(t, err)
		assert.True(t, exists)
		assert.Equal(t, 4, v)
	})
}

func TestRepository_DropAll(t *testing.T) {
	repo := memstore.NewRepository()

	{ // 1 - Write some data
		s1, err := repo.Store("s1")
		require.NoError(t, err)
		require.NoError(t, s1.Put([]byte{11}, 1))

		s2, err := repo.Store("s2")
		require.NoError(t, err)
		require.NoError(t, s2.Put([]byte{22}, 2))
	}

	{ // 2 - Data is still readable
		s1, err := repo.Store("s1")
		require.NoError(t, err)
		keys, err := s1.Keys()
		require.NoError(t, err)
		assert.Equal(t, [][]byte{{11}}, keys)

		s2, err := repo.Store("s2")
		require.NoError(t, err)
		keys, err = s2.Keys()
		require.NoError(t, err)
		assert.Equal(t, [][]byte{{22}}, keys)
	}

	// 3 - Drop all
	require.NoError(t, repo.DropAll())

	{ // 4 - Data most not be there anymore
		s1, err := repo.Store("s1")
		require.NoError(t, err)
		keys, err := s1.Keys()
		require.NoError(t, err)
		assert.Nil(t, keys)

		s2, err := repo.Store("s2")
		require.NoError(t, err)
		keys, err = s2.Keys()
		require.NoError(t, err)
		assert.Nil(t, keys)
	}
}
