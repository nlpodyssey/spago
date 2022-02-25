// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diskstore_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ store.Repository = &diskstore.Repository{}

func TestNewRepository(t *testing.T) {
	t.Run("invalid RepositoryMode", func(t *testing.T) {
		repo, err := diskstore.NewRepository(t.TempDir(), 42)
		assert.Error(t, err)
		assert.Nil(t, repo)
	})

	t.Run("ReadWriteMode - directory exists", func(t *testing.T) {
		repo, err := diskstore.NewRepository(t.TempDir(), diskstore.ReadWriteMode)
		require.NoError(t, err)
		assert.NotNil(t, repo)
		assert.NoError(t, repo.Close())
	})

	t.Run("ReadWriteMode - directory does not exist", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "foo")
		repo, err := diskstore.NewRepository(path, diskstore.ReadWriteMode)
		require.NoError(t, err)
		assert.NotNil(t, repo)
		assert.DirExists(t, path)
		assert.NoError(t, repo.Close())
	})

	t.Run("ReadWriteMode - directory and parents do not exist", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "foo", "bar", "baz")
		repo, err := diskstore.NewRepository(path, diskstore.ReadWriteMode)
		require.NoError(t, err)
		assert.NotNil(t, repo)
		assert.DirExists(t, path)
		assert.NoError(t, repo.Close())
	})

	t.Run("ReadWriteMode - directory creation error", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "foo")
		require.NoError(t, os.Mkdir(path, 0000))
		repo, err := diskstore.NewRepository(filepath.Join(path, "bar"), diskstore.ReadWriteMode)
		assert.Error(t, err)
		assert.Nil(t, repo)
	})

	t.Run("ReadOnlyMode - directory exists", func(t *testing.T) {
		repo, err := diskstore.NewRepository(t.TempDir(), diskstore.ReadOnlyMode)
		require.NoError(t, err)
		assert.NotNil(t, repo)
		assert.NoError(t, repo.Close())
	})

	t.Run("ReadOnlyMode - directory does not exist", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "foo")
		repo, err := diskstore.NewRepository(path, diskstore.ReadOnlyMode)
		assert.Error(t, err)
		assert.Nil(t, repo)
	})
}

func TestRepository_Store(t *testing.T) {
	t.Run("it returns a readable and writable Store in ReadWriteMode", func(t *testing.T) {
		repo, err := diskstore.NewRepository(t.TempDir(), diskstore.ReadWriteMode)
		require.NoError(t, err)
		defer func() { assert.NoError(t, repo.Close()) }()

		s, err := repo.Store("test-store")
		require.NoError(t, err)

		require.NoError(t, s.Put([]byte{1, 2}, []byte{11}))

		keys, err := s.Keys()
		require.NoError(t, err)
		assert.Equal(t, [][]byte{{1, 2}}, keys)
	})

	t.Run("it returns an error if a Store does not exist on disk in ReadOnlyMode", func(t *testing.T) {
		repo, err := diskstore.NewRepository(t.TempDir(), diskstore.ReadOnlyMode)
		require.NoError(t, err)
		defer func() { assert.NoError(t, repo.Close()) }()

		s, err := repo.Store("test-store")
		assert.Error(t, err)
		assert.Nil(t, s)
	})

	t.Run("it returns a non-writable existing Store in ReadOnlyMode", func(t *testing.T) {
		dir := t.TempDir()

		func() { // Preparation
			repo, err := diskstore.NewRepository(dir, diskstore.ReadWriteMode)
			require.NoError(t, err)
			defer func() { assert.NoError(t, repo.Close()) }()
			s, err := repo.Store("test-store")
			require.NoError(t, err)
			require.NoError(t, s.Put([]byte{1, 2}, []byte{3, 4}))
		}()

		repo, err := diskstore.NewRepository(dir, diskstore.ReadOnlyMode)
		require.NoError(t, err)
		defer func() { assert.NoError(t, repo.Close()) }()

		s, err := repo.Store("test-store")
		require.NoError(t, err)

		keys, err := s.Keys()
		require.NoError(t, err)
		assert.Equal(t, [][]byte{{1, 2}}, keys)

		// Write / delete operations must panic or return errors
		assert.Error(t, s.Put([]byte{6, 7}, []byte{8, 9}))
		assert.Panics(t, func() { _ = s.DropAll() })

		// Data should still be there
		keys, err = s.Keys()
		require.NoError(t, err)
		assert.Equal(t, [][]byte{{1, 2}}, keys)
	})

	t.Run("it returns memoized instances when called multiple times", func(t *testing.T) {
		repo, err := diskstore.NewRepository(t.TempDir(), diskstore.ReadWriteMode)
		require.NoError(t, err)
		defer func() { assert.NoError(t, repo.Close()) }()

		s1a, err := repo.Store("s1")
		require.NoError(t, err)

		s1b, err := repo.Store("s1")
		require.NoError(t, err)
		assert.Same(t, s1a, s1b)

		s2, err := repo.Store("s2")
		require.NoError(t, err)
		assert.NotSame(t, s1a, s2)

		err = repo.Close() // This also resets memoized values
		require.NoError(t, err)

		s1c, err := repo.Store("s1")
		require.NoError(t, err)
		assert.NotSame(t, s1a, s1c)
	})

	t.Run("data in each store is isolated", func(t *testing.T) {
		repo, err := diskstore.NewRepository(t.TempDir(), diskstore.ReadWriteMode)
		require.NoError(t, err)
		defer func() { assert.NoError(t, repo.Close()) }()

		// Insert in different stores

		a, err := repo.Store("")
		require.NoError(t, err)
		require.NoError(t, a.Put([]byte{11}, []byte{1}))

		b, err := repo.Store("b")
		require.NoError(t, err)
		require.NoError(t, b.Put([]byte{22}, []byte{2}))
		require.NoError(t, b.Put([]byte{33}, []byte{3}))

		c, err := repo.Store("c")
		require.NoError(t, err)
		require.NoError(t, c.Put([]byte{44}, []byte{4}))

		// Read from the stores

		var v []byte

		keys, err := a.Keys()
		require.NoError(t, err)
		assert.Equal(t, [][]byte{{11}}, keys)

		exists, err := a.Get([]byte{11}, &v)
		require.NoError(t, err)
		assert.True(t, exists)
		assert.Equal(t, []byte{1}, v)

		keys, err = b.Keys()
		require.NoError(t, err)
		assert.Len(t, keys, 2)
		assert.Contains(t, keys, []byte{22})
		assert.Contains(t, keys, []byte{33})

		exists, err = b.Get([]byte{22}, &v)
		require.NoError(t, err)
		assert.True(t, exists)
		assert.Equal(t, []byte{2}, v)

		exists, err = b.Get([]byte{33}, &v)
		require.NoError(t, err)
		assert.True(t, exists)
		assert.Equal(t, []byte{3}, v)

		keys, err = c.Keys()
		require.NoError(t, err)
		assert.Equal(t, [][]byte{{44}}, keys)

		exists, err = c.Get([]byte{44}, &v)
		require.NoError(t, err)
		assert.True(t, exists)
		assert.Equal(t, []byte{4}, v)
	})
}

func TestRepository_DropAll(t *testing.T) {
	t.Run("it removes all data in ReadWriteMode", func(t *testing.T) {
		dir := filepath.Join(t.TempDir(), "foo")

		func() { // 1 - Write some data
			repo, err := diskstore.NewRepository(dir, diskstore.ReadWriteMode)
			require.NoError(t, err)
			defer func() { assert.NoError(t, repo.Close()) }()

			s1, err := repo.Store("s1")
			require.NoError(t, err)
			require.NoError(t, s1.Put([]byte{11}, []byte{1}))

			s2, err := repo.Store("s2")
			require.NoError(t, err)
			require.NoError(t, s2.Put([]byte{22}, []byte{2}))
		}()

		func() { // 2 - Ensure data is still readable, then drop all
			repo, err := diskstore.NewRepository(dir, diskstore.ReadWriteMode)
			require.NoError(t, err)
			defer func() { assert.NoError(t, repo.Close()) }()

			s1, err := repo.Store("s1")
			keys, err := s1.Keys()
			require.NoError(t, err)
			assert.Equal(t, [][]byte{{11}}, keys)

			s2, err := repo.Store("s2")
			keys, err = s2.Keys()
			require.NoError(t, err)
			assert.Equal(t, [][]byte{{22}}, keys)

			require.NoError(t, repo.DropAll())
		}()

		// The directory must still exist, but must be empty
		assert.DirExists(t, dir)
		assertDirIsEmpty(t, dir)

		// 3 - Data most not be there anymore

		repo, err := diskstore.NewRepository(dir, diskstore.ReadWriteMode)
		require.NoError(t, err)
		defer func() { assert.NoError(t, repo.Close()) }()

		s1, err := repo.Store("s1")
		keys, err := s1.Keys()
		require.NoError(t, err)
		assert.Nil(t, keys)

		s2, err := repo.Store("s2")
		keys, err = s2.Keys()
		require.NoError(t, err)
		assert.Nil(t, keys)
	})

	t.Run("it returns an error in ReadOnlyMode", func(t *testing.T) {
		repo, err := diskstore.NewRepository(t.TempDir(), diskstore.ReadOnlyMode)
		require.NoError(t, err)
		defer func() { assert.NoError(t, repo.Close()) }()

		assert.Error(t, repo.DropAll())
	})
}

func assertDirIsEmpty(t *testing.T, path string) {
	t.Helper()

	f, err := os.Open(path)
	require.NoError(t, err)

	defer func() {
		assert.NoError(t, f.Close())
	}()

	infos, err := f.Readdir(0)
	require.NoError(t, err)
	assert.Empty(t, infos)
}
