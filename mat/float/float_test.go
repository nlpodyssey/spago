// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package float_test

import (
	"testing"

	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestFloat(t *testing.T) {
	t.Run("float32", testFloat[float32])
	t.Run("float64", testFloat[float64])
}

func testFloat[T float.DType](t *testing.T) {
	v := float.Interface[T](T(42))
	assert.Equal(t, float32(42), v.F32())
	assert.Equal(t, float64(42), v.F64())
}

func TestDTFloat(t *testing.T) {
	t.Run("it returns the correct value according to the type", func(t *testing.T) {
		f := fakeFloat{f32: 32, f64: 64}
		assert.Equal(t, float32(32), float.ValueOf[float32](f))
		assert.Equal(t, float64(64), float.ValueOf[float64](f))
	})

	t.Run("it panics with nil", func(t *testing.T) {
		assert.Panics(t, func() { float.ValueOf[float32](nil) })
		assert.Panics(t, func() { float.ValueOf[float64](nil) })
	})
}

type fakeFloat struct {
	f32 float32
	f64 float64
}

func (f fakeFloat) F32() float32 { return f.f32 }
func (f fakeFloat) F64() float64 { return f.f64 }
