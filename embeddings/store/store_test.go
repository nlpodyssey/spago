// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package store_test

import (
	"bytes"
	"encoding/gob"
	"testing"

	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPreventStoreMarshaling_MarshalBinary(t *testing.T) {
	st := store.PreventStoreMarshaling{
		Store: &TestingStore{name: "foo"},
	}

	data, err := st.MarshalBinary()
	assert.NoError(t, err)
	assert.Empty(t, data)
}

func TestPreventStoreMarshaling_UnmarshalBinary(t *testing.T) {
	t.Run("nil", func(t *testing.T) {
		s := new(store.PreventStoreMarshaling)
		assert.NoError(t, s.UnmarshalBinary(nil))
		assert.Nil(t, s.Store)
	})

	t.Run("empty slice", func(t *testing.T) {
		s := new(store.PreventStoreMarshaling)
		assert.NoError(t, s.UnmarshalBinary([]byte{}))
		assert.Nil(t, s.Store)
	})

	t.Run("non empty slice", func(t *testing.T) {
		s := new(store.PreventStoreMarshaling)
		assert.Error(t, s.UnmarshalBinary([]byte{1}))
		assert.Nil(t, s.Store)
	})
}

func TestGobEncoding(t *testing.T) {
	// Having already tested the marshaling methods, also testing that gob
	// works as expected is a little redundant.
	// However, since gob is the primary method used in spaGO to serialize all
	// models, it's better to be extra safe than sorry :-)

	type MyStruct struct {
		Foo store.Store
		Bar store.Store
	}

	var data []byte
	{
		ms := MyStruct{
			Foo: &TestingStore{name: "foo"},
			Bar: store.PreventStoreMarshaling{
				Store: &TestingStore{name: "bar"},
			},
		}
		var buf bytes.Buffer
		require.NoError(t, gob.NewEncoder(&buf).Encode(ms))
		data = buf.Bytes()
	}

	var ms MyStruct
	require.NoError(t, gob.NewDecoder(bytes.NewReader(data)).Decode(&ms))

	require.NotNil(t, ms.Foo)
	assert.Equal(t, "foo", ms.Foo.Name())
	require.NotNil(t, ms.Bar)
	assert.Nil(t, ms.Bar.(store.PreventStoreMarshaling).Store)
}

type TestingStore struct {
	name string
}

func init() {
	gob.Register(TestingStore{})
}

func (t TestingStore) Name() string                  { return t.name }
func (t TestingStore) DropAll() error                { panic("this should never be called") }
func (t TestingStore) Keys() ([][]byte, error)       { panic("this should never be called") }
func (t TestingStore) KeysCount() (int, error)       { panic("this should never be called") }
func (t TestingStore) Contains([]byte) (bool, error) { panic("this should never be called") }
func (t TestingStore) Put([]byte, any) error         { panic("this should never be called") }
func (t TestingStore) Get([]byte, any) (bool, error) { panic("this should never be called") }

func (t TestingStore) MarshalBinary() ([]byte, error) {
	return []byte(t.name), nil
}

func (t *TestingStore) UnmarshalBinary(data []byte) error {
	t.name = string(data)
	return nil
}
