// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"bytes"
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
