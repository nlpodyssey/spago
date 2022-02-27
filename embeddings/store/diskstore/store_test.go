// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package diskstore_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ store.Store = &diskstore.Store{}

func TestStore_Name(t *testing.T) {
	s, closeRepo := makeStore(t)
	defer closeRepo()

	assert.Equal(t, "test-store", s.Name())
}

func TestStore_DropAll(t *testing.T) {
	s, closeRepo := makeStore(t)
	defer closeRepo()

	require.NoError(t, s.Put([]byte{11}, []byte{1}))
	require.NoError(t, s.Put([]byte{22}, []byte{2}))

	// Data is available before deletion

	var v []byte

	exists, err := s.Get([]byte{11}, &v)
	require.NoError(t, err)
	assert.True(t, exists)
	assert.Equal(t, []byte{1}, v)

	exists, err = s.Get([]byte{22}, &v)
	require.NoError(t, err)
	assert.True(t, exists)
	assert.Equal(t, []byte{2}, v)

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
	s, closeRepo := makeStore(t)
	defer closeRepo()

	keys, err := s.Keys()
	require.NoError(t, err)
	assert.Nil(t, keys)

	require.NoError(t, s.Put([]byte{1}, []byte{11}))

	keys, err = s.Keys()
	require.NoError(t, err)
	assert.Equal(t, [][]byte{{1}}, keys)

	require.NoError(t, s.Put([]byte{2, 3}, []byte{22}))

	keys, err = s.Keys()
	require.NoError(t, err)
	assert.Len(t, keys, 2)
	assert.Contains(t, keys, []byte{1})
	assert.Contains(t, keys, []byte{2, 3})
}

func TestStore_KeysCount(t *testing.T) {
	s, closeRepo := makeStore(t)
	defer closeRepo()

	n, err := s.KeysCount()
	require.NoError(t, err)
	assert.Equal(t, 0, n)

	require.NoError(t, s.Put([]byte{1}, []byte{11}))

	n, err = s.KeysCount()
	require.NoError(t, err)
	assert.Equal(t, 1, n)

	require.NoError(t, s.Put([]byte{2}, []byte{22}))

	n, err = s.KeysCount()
	require.NoError(t, err)
	assert.Equal(t, 2, n)
}

func TestStore_Contains(t *testing.T) {
	s, closeRepo := makeStore(t)
	defer closeRepo()

	c, err := s.Contains([]byte{1, 2})
	require.NoError(t, err)
	assert.False(t, c)

	c, err = s.Contains([]byte{3})
	require.NoError(t, err)
	assert.False(t, c)

	require.NoError(t, s.Put([]byte{1, 2}, []byte{11}))

	c, err = s.Contains([]byte{1, 2})
	require.NoError(t, err)
	assert.True(t, c)

	c, err = s.Contains([]byte{3})
	require.NoError(t, err)
	assert.False(t, c)

	require.NoError(t, s.Put([]byte{3}, []byte{22}))

	c, err = s.Contains([]byte{1, 2})
	require.NoError(t, err)
	assert.True(t, c)

	c, err = s.Contains([]byte{3})
	require.NoError(t, err)
	assert.True(t, c)
}

func TestStore_PutAndGet(t *testing.T) {
	t.Run("simple []byte values", func(t *testing.T) {
		s, closeRepo := makeStore(t)
		defer closeRepo()

		var v []byte

		exists, err := s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.False(t, exists)

		require.NoError(t, s.Put([]byte{1, 2}, []byte{11}))

		exists, err = s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, []byte{11}, v)

		// Overwrite the same key
		require.NoError(t, s.Put([]byte{1, 2}, []byte{11, 22}))

		exists, err = s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, []byte{11, 22}, v)

		require.NoError(t, s.Put([]byte{3, 4}, []byte{7, 8, 9}))

		exists, err = s.Get([]byte{3, 4}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, []byte{7, 8, 9}, v)

		// Try an empty slice
		require.NoError(t, s.Put([]byte{5, 6}, []byte{}))

		exists, err = s.Get([]byte{5, 6}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Nil(t, v)

		// Try []byte(nil)
		require.NoError(t, s.Put([]byte{7, 8}, []byte(nil)))

		exists, err = s.Get([]byte{7, 8}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Nil(t, v)
	})

	t.Run("binary-marshalable value", func(t *testing.T) {
		s, closeRepo := makeStore(t)
		defer closeRepo()

		require.NoError(t, s.Put([]byte{1, 2}, binaryMarshalable{Foo: 11, Bar: 22}))

		var v binaryMarshalable
		exists, err := s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, binaryMarshalable{Foo: 11, Bar: 22}, v)
	})

	t.Run("binary-marshalable value - marshaling error", func(t *testing.T) {
		s, closeRepo := makeStore(t)
		defer closeRepo()

		err := s.Put([]byte{1, 2}, binaryMarshalable{MarshalingError: true})
		assert.ErrorIs(t, err, marshalingError)
	})

	t.Run("binary-marshalable value - unmarshaling error", func(t *testing.T) {
		s, closeRepo := makeStore(t)
		defer closeRepo()

		err := s.Put([]byte{1, 2}, binaryMarshalable{UnmarshalingError: true})

		var v binaryMarshalable
		exists, err := s.Get([]byte{1, 2}, &v)
		assert.ErrorIs(t, err, unmarshalingError)
		require.True(t, exists)
	})

	t.Run("gob-encodable value", func(t *testing.T) {
		s, closeRepo := makeStore(t)
		defer closeRepo()

		require.NoError(t, s.Put([]byte{1, 2}, gobEncodable{Foo: 11, Bar: 22}))

		var v gobEncodable
		exists, err := s.Get([]byte{1, 2}, &v)
		require.NoError(t, err)
		require.True(t, exists)
		assert.Equal(t, gobEncodable{Foo: 11, Bar: 22}, v)
	})

	t.Run("gob-encodable value - encoding error", func(t *testing.T) {
		s, closeRepo := makeStore(t)
		defer closeRepo()

		err := s.Put([]byte{1, 2}, gobEncodable{EncodingError: true})
		assert.ErrorIs(t, err, gobEncodingError)
	})

	t.Run("gob-encodable value - decoding error", func(t *testing.T) {
		s, closeRepo := makeStore(t)
		defer closeRepo()

		err := s.Put([]byte{1, 2}, gobEncodable{DecodingError: true})

		var v gobEncodable
		exists, err := s.Get([]byte{1, 2}, &v)
		assert.ErrorIs(t, err, gobDecodingError)
		require.True(t, exists)
	})

	t.Run("non-encodable type", func(t *testing.T) {
		s, closeRepo := makeStore(t)
		defer closeRepo()

		err := s.Put([]byte{1, 2}, "foo")
		assert.Error(t, err)
	})

	t.Run("non-decodable parameter type", func(t *testing.T) {
		s, closeRepo := makeStore(t)
		defer closeRepo()

		err := s.Put([]byte{1, 2}, []byte{11, 22})

		var v string
		exists, err := s.Get([]byte{1, 2}, &v)
		assert.Error(t, err)
		require.True(t, exists)
	})
}

var (
	marshalingError   = errors.New("marshaling error")
	unmarshalingError = errors.New("unmarshaling error")
	gobEncodingError  = errors.New("gob encoding error")
	gobDecodingError  = errors.New("gob decoding error")
)

type binaryMarshalable struct {
	Foo byte
	Bar byte
	// If set to true, MarshalBinary will raise marshalingError
	MarshalingError bool
	// If set to true, UnmarshalBinary will raise unmarshalingError
	UnmarshalingError bool
}

func (bm binaryMarshalable) MarshalBinary() ([]byte, error) {
	if bm.MarshalingError {
		return nil, marshalingError
	}
	data := []byte{0, bm.Foo, bm.Bar}
	if bm.UnmarshalingError {
		data[0] = 1
	}
	return data, nil
}

func (bm *binaryMarshalable) UnmarshalBinary(data []byte) error {
	if len(data) != 3 {
		return fmt.Errorf("expected data len 3, actual %d", len(data))
	}
	if data[0] == 1 {
		return unmarshalingError
	}
	bm.Foo = data[1]
	bm.Bar = data[2]
	return nil
}

type gobEncodable struct {
	Foo byte
	Bar byte
	// If set to true, GobEncode will raise gobEncodingError
	EncodingError bool
	// If set to true, GobDecode will raise gobDecodingError
	DecodingError bool
}

func (ge gobEncodable) GobEncode() ([]byte, error) {
	if ge.EncodingError {
		return nil, gobEncodingError
	}
	data := []byte{0, ge.Foo, ge.Bar}
	if ge.DecodingError {
		data[0] = 1
	}
	return data, nil
}

func (ge *gobEncodable) GobDecode(data []byte) error {
	if len(data) != 3 {
		return fmt.Errorf("expected data len 3, actual %d", len(data))
	}
	if data[0] == 1 {
		return gobDecodingError
	}
	ge.Foo = data[1]
	ge.Bar = data[2]
	return nil
}

func makeStore(t *testing.T) (_ store.Store, closeRepo func()) {
	t.Helper()

	repo, err := diskstore.NewRepository(t.TempDir(), diskstore.ReadWriteMode)
	require.NoError(t, err)

	s, err := repo.Store("test-store")
	require.NoError(t, err)

	closeRepo = func() { assert.NoError(t, repo.Close()) }
	return s, closeRepo
}
