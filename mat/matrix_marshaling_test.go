// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMatrixMarshaling(t *testing.T) {
	t.Run("Dense float32", testMatrixMarshalingDense[float32])
	t.Run("Dense float64", testMatrixMarshalingDense[float64])

	t.Run("nil", func(t *testing.T) {

		var buf bytes.Buffer
		err := MarshalBinaryMatrix(nil, &buf)
		require.NoError(t, err)

		m, err := UnmarshalBinaryMatrix(&buf)
		require.NoError(t, err)
		assert.Nil(t, m)
	})
}

func testMatrixMarshalingDense[T float.DType](t *testing.T) {
	testCases := []Matrix{
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
			err := MarshalBinaryMatrix(tc, &buf)
			require.NoError(t, err)

			m, err := UnmarshalBinaryMatrix(&buf)
			require.NoError(t, err)
			assert.Equal(t, tc.Rows(), m.Rows())
			assert.Equal(t, tc.Columns(), m.Columns())
			assert.Equal(t, tc.Data(), m.Data())
		})
	}
}
