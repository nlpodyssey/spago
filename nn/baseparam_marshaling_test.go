// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"encoding/gob"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
		assert.Equal(t, mat.Float(T(12)), decodedParam.Value().Scalar())

		payload := decodedParam.Payload()
		assert.NotNil(t, payload)
		assert.Equal(t, 42, payload.Label)
		assert.NotEmpty(t, payload.Data)
		assert.Equal(t, mat.Float(T(34)), payload.Data[0].Scalar())
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

func TestParamInterfaceBinaryMarshaling(t *testing.T) {
	t.Run("float32", testParamInterfaceBinaryMarshaling[float32])
	t.Run("float64", testParamInterfaceBinaryMarshaling[float64])
}

func testParamInterfaceBinaryMarshaling[T mat.DType](t *testing.T) {
	t.Run("simple case", func(t *testing.T) {
		buf := new(bytes.Buffer)

		paramToEncode := NewParam[T](mat.NewScalar[T](42)).(*BaseParam[T])
		err := MarshalBinaryParam(paramToEncode, buf)
		require.Nil(t, err)

		decodedParam, err := UnmarshalBinaryParam[T](buf)
		require.Nil(t, err)
		require.NotNil(t, decodedParam)
		assert.Equal(t, mat.Float(T(42)), decodedParam.Value().Scalar())
	})

	t.Run("nil", func(t *testing.T) {
		buf := new(bytes.Buffer)

		var paramToEncode *BaseParam[T] = nil
		err := MarshalBinaryParam(paramToEncode, buf)
		require.Nil(t, err)

		decodedParam, err := UnmarshalBinaryParam[T](buf)
		require.Nil(t, err)
		assert.Nil(t, decodedParam)
	})
}
