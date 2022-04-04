// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"bytes"
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"testing"
)

func TestMatrixMarshaling(t *testing.T) {
	t.Run("Dense float32", testMatrixMarshalingDense[float32])
	t.Run("Dense float64", testMatrixMarshalingDense[float64])

	t.Run("nil float32", testMatrixMarshalingNil[float32])
	t.Run("nil float64", testMatrixMarshalingNil[float64])

	t.Run("wrong types Dense float32-float64", testMatrixMarshalingDenseWrongType[float32, float64])
	t.Run("wrong types Dense float64-float32", testMatrixMarshalingDenseWrongType[float64, float32])
}

func testMatrixMarshalingDense[T DType](t *testing.T) {
	testCases := []Matrix[T]{
		NewEmptyDense[T](0, 0),
		NewEmptyDense[T](0, 1),
		NewEmptyDense[T](1, 0),
		NewDense[T](1, 1, []T{1}),
		NewDense[T](1, 2, []T{1, 2}),
		NewDense[T](2, 1, []T{1, 2}),
		NewDense[T](2, 2, []T{-1, 2, -3, 4}),
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.Rows(), tc.Columns()), func(t *testing.T) {
			var buf bytes.Buffer
			err := MarshalBinaryMatrix[T](tc, &buf)
			require.NoError(t, err)

			m, err := UnmarshalBinaryMatrix[T](&buf)
			require.NoError(t, err)
			assert.Equal(t, tc.Rows(), m.Rows())
			assert.Equal(t, tc.Columns(), m.Columns())
			assert.Equal(t, tc.Data(), m.Data())
			assert.Equal(t, denseFlag(0), m.(*Dense[T]).flags)
		})
	}
}

func testMatrixMarshalingNil[T DType](t *testing.T) {
	var buf bytes.Buffer
	err := MarshalBinaryMatrix[T](nil, &buf)
	require.NoError(t, err)

	m, err := UnmarshalBinaryMatrix[T](&buf)
	require.NoError(t, err)
	assert.Nil(t, m)
}

func testMatrixMarshalingDenseWrongType[T1, T2 DType](t *testing.T) {
	var d Matrix[T1] = NewScalar[T1](42)
	var buf bytes.Buffer
	err := MarshalBinaryMatrix[T1](d, &buf)
	require.NoError(t, err)

	m, err := UnmarshalBinaryMatrix[T2](&buf)
	assert.Error(t, err)
	assert.Nil(t, m)
}
