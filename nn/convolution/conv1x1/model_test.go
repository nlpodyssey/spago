// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conv1x1

import (
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModel_Forward(t *testing.T) {
	t.Run("float32", testModelForward[float32])
	t.Run("float64", testModelForward[float64])
}

func testModelForward[T float.DType](t *testing.T) {
	t.Run("input 3, output 2", func(t *testing.T) {
		model := New[T](Config{
			InputChannels:  3,
			OutputChannels: 2,
		})

		require.Equal(t, 2, model.W.Value().Shape()[0])
		require.Equal(t, 3, model.W.Value().Shape()[1])
		require.Equal(t, 2, model.B.Value().Shape()[0])
		require.Equal(t, 1, model.B.Value().Shape()[1])

		mat.SetData[T](model.B.Value(), []T{0.1, 0.2})
		mat.SetData[T](model.W.Value(), []T{
			1, 2, 3,
			4, 5, 6,
		})

		xs := []ag.DualValue{
			mat.NewDense[T](mat.WithBacking([]T{1, 2, 4, 0, -1})),
			mat.NewDense[T](mat.WithBacking([]T{1, 3, 3, 0, -1})),
			mat.NewDense[T](mat.WithBacking([]T{1, 4, 2, 0, -1})),
		}
		ys := model.Forward(xs...)
		require.Len(t, ys, 2)
		require.True(t, mat.IsVector(ys[0].Value()))
		require.Equal(t, 5, ys[0].Value().Size())
		require.True(t, mat.IsVector(ys[1].Value()))
		require.Equal(t, 5, ys[1].Value().Size())
		assert.InDeltaSlice(t, []T{6.1, 20.1, 16.1, 0.1, -5.9}, ys[0].Value().Data(), 0.001)
		assert.InDeltaSlice(t, []T{15.2, 47.2, 43.2, 0.2, -14.8}, ys[1].Value().Data(), 0.001)
	})

	t.Run("input 4, output 3", func(t *testing.T) {
		model := New[T](Config{
			InputChannels:  4,
			OutputChannels: 3,
		})

		mat.SetData[T](model.B.Value(), []T{0.6, 0.5, 0.7})
		mat.SetData[T](model.W.Value(), []T{
			0.1, 0.2, 0.3, 0.4,
			0.5, 0.6, 0.7, 0.8,
			0.9, 0.8, 0.7, 0.6,
		})

		xs := []ag.DualValue{
			mat.NewDense[T](mat.WithBacking([]T{0.2, 0.9, 0.1})),
			mat.NewDense[T](mat.WithBacking([]T{0.4, 0.7, 0.1})),
			mat.NewDense[T](mat.WithBacking([]T{0.6, 0.5, 0.1})),
			mat.NewDense[T](mat.WithBacking([]T{0.8, 0.3, 0.1})),
		}
		ys := model.Forward(xs...)
		assert.InDeltaSlice(t, []T{1.2, 1.1, 0.7}, ys[0].Value().Data(), 0.001)
		assert.InDeltaSlice(t, []T{1.9, 1.96, 0.76}, ys[1].Value().Data(), 0.001)
		assert.InDeltaSlice(t, []T{2.1, 2.6, 1}, ys[2].Value().Data(), 0.001)
	})
}
