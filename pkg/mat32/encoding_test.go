// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"bytes"
	"encoding/gob"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestDense_Gob(t *testing.T) {
	matrixToEncode := NewDense(2, 3, []Float{
		1, 2, 3,
		4, 5, 6,
	})

	var buf bytes.Buffer

	enc := gob.NewEncoder(&buf)
	err := enc.Encode(matrixToEncode)
	require.Nil(t, err)

	var decodedMatrix *Dense

	dec := gob.NewDecoder(&buf)
	err = dec.Decode(&decodedMatrix)
	require.Nil(t, err)

	assert.NotNil(t, decodedMatrix)
	assert.Equal(t, 2, decodedMatrix.rows)
	assert.Equal(t, 3, decodedMatrix.cols)
	assert.Equal(t, 6, decodedMatrix.size)
	assert.Equal(t, []Float{1, 2, 3, 4, 5, 6}, decodedMatrix.data)
	assert.Nil(t, decodedMatrix.viewOf)
	assert.False(t, decodedMatrix.fromPool)
}

func TestSparse_Gob(t *testing.T) {
	matrixToEncode := NewSparse(2, 3, []Float{
		1, 0, 3,
		0, 5, 0,
	})

	var buf bytes.Buffer

	enc := gob.NewEncoder(&buf)
	err := enc.Encode(matrixToEncode)
	require.Nil(t, err)

	var decodedMatrix *Dense

	dec := gob.NewDecoder(&buf)
	err = dec.Decode(&decodedMatrix)
	require.Nil(t, err)

	assert.NotNil(t, decodedMatrix)
	assert.Equal(t, 2, decodedMatrix.Rows())
	assert.Equal(t, 3, decodedMatrix.Columns())
	assert.Equal(t, []Float{1, 0, 3, 0, 5, 0}, decodedMatrix.Data())
}

func TestMatrixBinaryMarshaling(t *testing.T) {
	t.Run("Dense matrix", func(t *testing.T) {
		var matrixToEncode Matrix = NewScalar(42)

		buf := new(bytes.Buffer)
		err := MarshalBinaryMatrix(matrixToEncode, buf)
		require.Nil(t, err)

		decodedMatrix, err := UnmarshalBinaryMatrix(buf)
		require.Nil(t, err)
		require.NotNil(t, decodedMatrix)
		require.IsType(t, &Dense{}, decodedMatrix)
		require.Equal(t, Float(42), decodedMatrix.Scalar())
	})

	t.Run("Sparse matrix", func(t *testing.T) {
		var matrixToEncode Matrix = NewVecSparse([]Float{42})

		buf := new(bytes.Buffer)
		err := MarshalBinaryMatrix(matrixToEncode, buf)
		require.Nil(t, err)

		decodedMatrix, err := UnmarshalBinaryMatrix(buf)
		require.Nil(t, err)
		require.NotNil(t, decodedMatrix)
		require.IsType(t, &Sparse{}, decodedMatrix)
		require.Equal(t, Float(42), decodedMatrix.Scalar())
	})

	t.Run("nil", func(t *testing.T) {
		var matrixToEncode Matrix = nil

		buf := new(bytes.Buffer)
		err := MarshalBinaryMatrix(matrixToEncode, buf)
		require.Nil(t, err)

		decodedMatrix, err := UnmarshalBinaryMatrix(buf)
		require.Nil(t, err)
		require.Nil(t, decodedMatrix)
	})
}
