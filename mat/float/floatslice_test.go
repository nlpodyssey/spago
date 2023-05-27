// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package float_test

import (
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestFloatSlice(t *testing.T) {
	t.Run("float32", testFloatSlice[float32])
	t.Run("float64", testFloatSlice[float64])
}

func testFloatSlice[T float.DType](t *testing.T) {
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
			v := float.Make[T](tc.v...)
			assert.Equal(t, tc.f32, v.F32())
			assert.Equal(t, tc.f64, v.F64())
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
				assert.Equal(t, tc.f32, float.SliceValueOf[float32](tc))
				assert.Equal(t, tc.f64, float.SliceValueOf[float64](tc))
			})
		}
	})

	t.Run("it panics with nil", func(t *testing.T) {
		assert.Panics(t, func() { float.SliceValueOf[float32](nil) })
		assert.Panics(t, func() { float.SliceValueOf[float64](nil) })
	})
}

func TestFloatSliceImplementation_Len(t *testing.T) {
	t.Run("float32", testFloatSliceImplLen[float32])
	t.Run("float64", testFloatSliceImplLen[float64])
}

func testFloatSliceImplLen[T float.DType](t *testing.T) {
	testCases := []struct {
		v []T
		l int
	}{
		{nil, 0},
		{[]T{}, 0},
		{[]T{1}, 1},
		{[]T{1, 2}, 2},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%#v", tc.v), func(t *testing.T) {
			fs := float.Make[T](tc.v...)
			assert.Equal(t, tc.l, fs.Len())
		})
	}
}

func TestFloatSliceImplementation_Equals(t *testing.T) {
	t.Run("float32-float32", testFloatSliceImplEquals[float32, float32])
	t.Run("float32-float64", testFloatSliceImplEquals[float32, float64])
	t.Run("float64-float32", testFloatSliceImplEquals[float64, float32])
	t.Run("float64-float64", testFloatSliceImplEquals[float64, float64])
}

func testFloatSliceImplEquals[A, B float.DType](t *testing.T) {
	testCases := []struct {
		a []A
		b []B
		e bool
	}{
		{nil, nil, true},
		{nil, []B{}, true},
		{[]A{}, []B{}, true},
		{[]A{1}, []B{1}, true},
		{[]A{1, 2}, []B{1, 2}, true},
		{[]A{1}, nil, false},
		{[]A{1}, []B{}, false},
		{[]A{1}, []B{2}, false},
		{[]A{1}, []B{1, 2}, false},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%#v equals %#v", tc.a, tc.b), func(t *testing.T) {
			a := float.Make[A](tc.a...)
			b := float.Make[B](tc.b...)
			assert.Equal(t, tc.e, a.Equals(b))
			assert.Equal(t, tc.e, b.Equals(a))
		})
	}
}

func TestFloatSliceImplementation_InDelta(t *testing.T) {
	t.Run("float32-float32", testFloatSliceImplInDelta[float32, float32])
	t.Run("float32-float64", testFloatSliceImplInDelta[float32, float64])
	t.Run("float64-float32", testFloatSliceImplInDelta[float64, float32])
	t.Run("float64-float64", testFloatSliceImplInDelta[float64, float64])
}

func testFloatSliceImplInDelta[A, B float.DType](t *testing.T) {
	testCases := []struct {
		a []A
		b []B
		d float64
		e bool
	}{
		{nil, nil, 0, true},
		{nil, []B{}, 0, true},
		{[]A{}, []B{}, 0, true},
		{[]A{1}, []B{1}, 0, true},
		{[]A{1, 2}, []B{1, 2}, 0, true},

		{nil, nil, 1, true},
		{nil, []B{}, 1, true},
		{[]A{}, []B{}, 1, true},
		{[]A{1}, []B{1}, 1, true},
		{[]A{1, 2}, []B{1, 2}, 1, true},

		{[]A{1}, nil, 0, false},
		{[]A{1}, []B{}, 0, false},
		{[]A{1}, []B{2}, 0, false},
		{[]A{1}, []B{1, 2}, 0, false},

		{[]A{1}, nil, 1, false},
		{[]A{1}, []B{}, 1, false},
		{[]A{1}, []B{3}, 1, false},
		{[]A{1}, []B{1, 3}, 1, false},

		{[]A{1}, []B{2}, 1, true},
		{[]A{1, 2}, []B{0, 1}, 1, true},
	}
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%#v equals %#v", tc.a, tc.b), func(t *testing.T) {
			a := float.Make[A](tc.a...)
			b := float.Make[B](tc.b...)
			assert.Equal(t, tc.e, a.InDelta(b, tc.d))
			assert.Equal(t, tc.e, b.InDelta(a, tc.d))
		})
	}
}

type fakeFloatSlice struct {
	f32 []float32
	f64 []float64
	ln  int
}

func (f fakeFloatSlice) F32() []float32                    { return f.f32 }
func (f fakeFloatSlice) F64() []float64                    { return f.f64 }
func (f fakeFloatSlice) BitSize() int                      { return -1 }
func (f fakeFloatSlice) Len() int                          { return f.ln }
func (f fakeFloatSlice) Equals(float.Slice) bool           { panic("not implemented") }
func (f fakeFloatSlice) InDelta(float.Slice, float64) bool { panic("not implemented") }
