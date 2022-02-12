// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/utils/kvdb"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"os"
	"testing"
)

func TestParam_Gob(t *testing.T) {
	t.Run("float32", testParamGob[float32])
	t.Run("float64", testParamGob[float64])
}

func testParamGob[T mat.DType](t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		var buf bytes.Buffer

		paramToEncode := NewParam[T](mat.NewScalar[T](12))
		paramToEncode.SetPayload(&Payload[T]{
			Label: 42,
			Data:  []mat.Matrix[T]{mat.NewScalar[T](34)},
		})

		err := gob.NewEncoder(&buf).Encode(&paramToEncode)
		require.Nil(t, err)

		var decodedParam Param[T]

		err = gob.NewDecoder(&buf).Decode(&decodedParam)
		require.Nil(t, err)
		require.NotNil(t, decodedParam)
		require.NotNil(t, decodedParam.Value())
		assert.Equal(t, T(12), decodedParam.Value().Scalar())

		payload := decodedParam.Payload()
		assert.NotNil(t, payload)
		assert.Equal(t, 42, payload.Label)
		assert.NotEmpty(t, payload.Data)
		assert.Equal(t, T(34), payload.Data[0].Scalar())
	})

	t.Run("nil value and payload", func(t *testing.T) {
		var buf bytes.Buffer

		paramToEncode := NewParam[T](nil)

		err := gob.NewEncoder(&buf).Encode(&paramToEncode)
		require.Nil(t, err)

		var decodedParam Param[T]
		err = gob.NewDecoder(&buf).Decode(&decodedParam)
		require.Nil(t, err)
		require.NotNil(t, decodedParam)
		assert.Nil(t, decodedParam.Value())
		assert.Nil(t, decodedParam.Payload())
	})
}

func TestParam_Storage(t *testing.T) {
	t.Run("float32", testParamStorage[float32])
	t.Run("float64", testParamStorage[float64])
}

func testParamStorage[T mat.DType](t *testing.T) {
	dir, err := os.MkdirTemp("", "spago-kvdb-test-")
	require.Nil(t, err)
	defer os.RemoveAll(dir)

	storage := kvdb.NewDefaultKeyValueDB(kvdb.Config{Path: dir, ReadOnly: false, ForceNew: true})
	defer storage.Close()
	p := NewParam[T](mat.NewScalar[T](123), SetStorage[T](storage))
	p.SetName("foo")
	payload := NewPayload[T]()
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

	decodedParam, err := UnmarshalBinaryParam[T](bytes.NewReader(value))
	require.Nil(t, err)
	require.NotNil(t, decodedParam)
	require.Equal(t, T(123), decodedParam.Value().Scalar())
	require.Equal(t, payload, decodedParam.Payload())
}

func TestParamInterfaceBinaryMarshaling(t *testing.T) {
	t.Run("float32", testParamInterfaceBinaryMarshaling[float32])
	t.Run("float64", testParamInterfaceBinaryMarshaling[float64])
}

func testParamInterfaceBinaryMarshaling[T mat.DType](t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		buf := new(bytes.Buffer)

		paramToEncode := NewParam[T](mat.NewScalar[T](42))
		err := MarshalBinaryParam(paramToEncode, buf)
		require.Nil(t, err)

		decodedParam, err := UnmarshalBinaryParam[T](buf)
		require.Nil(t, err)
		require.NotNil(t, decodedParam)
		assert.Equal(t, T(42), decodedParam.Value().Scalar())
	})

	t.Run("nil", func(t *testing.T) {
		buf := new(bytes.Buffer)

		var paramToEncode Param[T] = nil
		err := MarshalBinaryParam(paramToEncode, buf)
		require.Nil(t, err)

		decodedParam, err := UnmarshalBinaryParam[T](buf)
		require.Nil(t, err)
		assert.Nil(t, decodedParam)
	})
}
