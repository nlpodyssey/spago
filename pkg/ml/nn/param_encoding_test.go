// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/utils/kvdb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"io/ioutil"
	"os"
	"testing"
)

func TestParam_Gob(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		var buf bytes.Buffer

		paramToEncode := NewParam(mat.NewScalar(12))
		paramToEncode.SetPayload(&Payload{
			Label: 42,
			Data:  []mat.Matrix{mat.NewScalar(34)},
		})

		err := gob.NewEncoder(&buf).Encode(&paramToEncode)
		require.Nil(t, err)

		var decodedParam Param

		err = gob.NewDecoder(&buf).Decode(&decodedParam)
		require.Nil(t, err)
		require.NotNil(t, decodedParam)
		require.NotNil(t, decodedParam.Value())
		assert.Equal(t, mat.Float(12), decodedParam.Value().Scalar())

		payload := decodedParam.Payload()
		assert.NotNil(t, payload)
		assert.Equal(t, 42, payload.Label)
		assert.NotEmpty(t, payload.Data)
		assert.Equal(t, mat.Float(34), payload.Data[0].Scalar())
	})

	t.Run("nil value and payload", func(t *testing.T) {
		var buf bytes.Buffer

		paramToEncode := NewParam(nil)

		err := gob.NewEncoder(&buf).Encode(&paramToEncode)
		require.Nil(t, err)

		var decodedParam Param
		err = gob.NewDecoder(&buf).Decode(&decodedParam)
		require.Nil(t, err)
		require.NotNil(t, decodedParam)
		assert.Nil(t, decodedParam.Value())
		assert.Nil(t, decodedParam.Payload())
	})
}

func TestParam_Storage(t *testing.T) {
	dir, err := ioutil.TempDir("", "spago-kvdb-test-")
	require.Nil(t, err)
	defer os.RemoveAll(dir)

	storage := kvdb.NewDefaultKeyValueDB(kvdb.Config{Path: dir, ReadOnly: false, ForceNew: true})
	defer storage.Close()
	p := NewParam(mat.NewScalar(123), SetStorage(storage))
	p.SetName("foo")
	payload := NewPayload()
	payload.Label = 42

	// Just run an operation which will update the storage
	p.SetPayload(payload)

	keys, err := storage.Keys()
	require.Nil(t, err)
	assert.Equal(t, []string{"foo"}, keys)

	value, ok, err := storage.Get([]byte("foo"))
	require.Nil(t, err)
	require.True(t, ok)
	assert.NotEmpty(t, value)

	decodedParam, err := UnmarshalBinaryParam(bytes.NewReader(value))
	require.Nil(t, err)
	require.NotNil(t, decodedParam)
	require.Equal(t, mat.Float(123), decodedParam.Value().Scalar())
	require.Equal(t, payload, decodedParam.Payload())
}

func TestParamInterfaceBinaryMarshaling(t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		buf := new(bytes.Buffer)

		paramToEncode := NewParam(mat.NewScalar(42))
		err := MarshalBinaryParam(paramToEncode, buf)
		require.Nil(t, err)

		decodedParam, err := UnmarshalBinaryParam(buf)
		require.Nil(t, err)
		require.NotNil(t, decodedParam)
		assert.Equal(t, mat.Float(42), decodedParam.Value().Scalar())
	})

	t.Run("nil", func(t *testing.T) {
		buf := new(bytes.Buffer)

		var paramToEncode Param = nil
		err := MarshalBinaryParam(paramToEncode, buf)
		require.Nil(t, err)

		decodedParam, err := UnmarshalBinaryParam(buf)
		require.Nil(t, err)
		assert.Nil(t, decodedParam)
	})
}
