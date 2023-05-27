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
		NewDense[T](WithShape(0, 0)),
		NewDense[T](WithShape(0, 1)),
		NewDense[T](WithShape(1, 0)),
		NewDense[T](WithShape(1, 1), WithBacking([]T{1})),
		NewDense[T](WithShape(1, 2), WithBacking([]T{1, 2})),
		NewDense[T](WithShape(2, 1), WithBacking([]T{1, 2})),
		NewDense[T](WithShape(2, 2), WithBacking([]T{-1, 2, -3, 4})),
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.Shape()[0], tc.Shape()[1]), func(t *testing.T) {
			var buf bytes.Buffer
			err := MarshalBinaryMatrix(tc, &buf)
			require.NoError(t, err)

			m, err := UnmarshalBinaryMatrix(&buf)
			require.NoError(t, err)
			assert.Equal(t, tc.Shape()[0], m.Shape()[0])
			assert.Equal(t, tc.Shape()[1], m.Shape()[1])
			assert.Equal(t, tc.Data(), m.Data())
		})
	}
}
