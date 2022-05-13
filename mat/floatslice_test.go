// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat_test

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"

	"github.com/nlpodyssey/spago/mat"
)

func TestFloatSlice(t *testing.T) {
	t.Run("float32", testFloatSlice[float32])
	t.Run("float64", testFloatSlice[float64])
}

func testFloatSlice[T mat.DType](t *testing.T) {
	testCases := []struct {
		v   []T
		f32 []float32
		f64 []float64
	}{
		{nil, nil, nil},
		{[]T{}, []float32{}, []float64{}},
		{[]T{1, 2}, []float32{1, 2}, []float64{1, 2}},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%#v", tc.v), func(t *testing.T) {
			v := mat.FloatSlice[T](tc.v)
			assert.Equal(t, tc.f32, v.Float32())
			assert.Equal(t, tc.f64, v.Float64())
		})
	}
}

func TestDTFloatSlice(t *testing.T) {
	t.Run("it returns the correct value according to the type", func(t *testing.T) {
		testCases := []fakeFloatSlice{
			{f32: nil, f64: nil},
			{f32: []float32{}, f64: []float64{}},
			{f32: []float32{1, 2}, f64: []float64{3, 4}},
		}

		for _, tc := range testCases {
			t.Run(fmt.Sprintf("%+v", tc), func(t *testing.T) {
				assert.Equal(t, tc.f32, mat.DTFloatSlice[float32](tc))
				assert.Equal(t, tc.f64, mat.DTFloatSlice[float64](tc))
			})
		}
	})

	t.Run("it panics with nil", func(t *testing.T) {
		assert.Panics(t, func() { mat.DTFloatSlice[float32](nil) })
		assert.Panics(t, func() { mat.DTFloatSlice[float64](nil) })
	})
}

type fakeFloatSlice struct {
	f32 []float32
	f64 []float64
}

func (f fakeFloatSlice) Float32() []float32 { return f.f32 }
func (f fakeFloatSlice) Float64() []float64 { return f.f64 }
