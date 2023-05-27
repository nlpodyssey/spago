// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDense_Marshaling(t *testing.T) {
	t.Run("float32", testDenseMarshaling[float32])
	t.Run("float64", testDenseMarshaling[float64])

	t.Run("wrong types float32-float64", testDenseMarshalingWrongType[float32, float64])
	t.Run("wrong types float64-float32", testDenseMarshalingWrongType[float64, float32])

	t.Run("gob encoding", func(t *testing.T) {
		type MyType struct {
			A Matrix
			B Matrix
		}

		x := MyType{
			A: NewDense[float32](WithShape(2, 2), WithBacking([]float32{1, 2, 3, 4})),
			B: NewDense[float64](WithShape(2, 2), WithBacking([]float64{5, 6, 7, 8})),
		}
		var buf bytes.Buffer

		enc := gob.NewEncoder(&buf)
		err := enc.Encode(x)
		require.Nil(t, err)

		var y MyType

		dec := gob.NewDecoder(&buf)
		err = dec.Decode(&y)
		require.Nil(t, err)

		assert.IsType(t, &Dense[float32]{}, y.A)
		assert.Equal(t, x.A.Shape()[0], y.A.Shape()[0])
		assert.Equal(t, x.A.Shape()[1], y.A.Shape()[1])
		assert.Equal(t, x.A.Data(), y.A.Data())

		assert.IsType(t, &Dense[float64]{}, y.B)
		assert.Equal(t, x.B.Shape()[0], y.B.Shape()[0])
		assert.Equal(t, x.B.Shape()[1], y.B.Shape()[1])
		assert.Equal(t, x.B.Data(), y.B.Data())
	})
}

func testDenseMarshaling[T float.DType](t *testing.T) {
	testCases := []*Dense[T]{
		NewDense[T](WithShape(0, 0)),
		NewDense[T](WithShape(0, 1)),
		NewDense[T](WithShape(1, 0)),
		NewDense[T](WithShape(1, 1), WithBacking([]T{1})),
		NewDense[T](WithShape(1, 2), WithBacking([]T{1, 2})),
		NewDense[T](WithShape(2, 1), WithBacking([]T{1, 2})),
		NewDense[T](WithShape(2, 2), WithBacking([]T{-1, 2, -3, 4})),
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%d x %d", tc.shape[0], tc.shape[1]), func(t *testing.T) {
			data, err := tc.MarshalBinary()
			require.NoError(t, err)

			y := new(Dense[T])
			err = y.UnmarshalBinary(data)
			require.NoError(t, err)
			assert.Equal(t, tc.shape[0], y.shape[0])
			assert.Equal(t, tc.shape[1], y.shape[1])
			assert.Equal(t, tc.data, y.data)
		})
	}
}

func testDenseMarshalingWrongType[T1, T2 float.DType](t *testing.T) {
	d := Scalar[T1](42)
	data, err := d.MarshalBinary()
	require.NoError(t, err)

	y := new(Dense[T2])
	err = y.UnmarshalBinary(data)
	assert.Error(t, err)
}
