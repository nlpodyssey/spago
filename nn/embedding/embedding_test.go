// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embedding_test

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn/embedding"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmbedding_Value(t *testing.T) {
	type T = float32

	m := embedding.New[T](2, 3)

	e1, _ := m.Embedding(0)
	e2, _ := m.Embedding(0)

	assert.NotNil(t, e1.Value())
	assert.NotNil(t, e2.Value())

	// Set a value for the first time
	e1.ReplaceValue(mat.NewDense[T](mat.WithBacking([]T{10, 20, 30})))

	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{10, 20, 30})), e1.Value())
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{10, 20, 30})), e2.Value())

	// Apply delta
	e1.ApplyDelta(mat.NewDense[T](mat.WithBacking([]T{1, 2, 3})))

	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{9, 18, 27})), e1.Value())
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{9, 18, 27})), e2.Value())

	// Set value to nil (weird corner case, but possible)

	e1.ReplaceValue(nil)

	assert.Nil(t, e1.Value())
	assert.Nil(t, e2.Value())

	assert.Panics(t, func() {
		e1.ApplyDelta(mat.NewDense[T](mat.WithBacking([]T{1, 2, 3})))
	}, "cannot apply delta to embedding with nil value")
}

func TestEmbedding_ReplaceValue(t *testing.T) {
	// it must clear grad and payload

	type T = float32

	m := embedding.New[T](1, 3)

	e, _ := m.Embedding(0)

	payload := []mat.Matrix{
		mat.NewDense[T](mat.WithBacking([]T{11, 22, 33})),
	}

	e.ReplaceValue(mat.NewDense[T](mat.WithBacking([]T{1, 2, 3})))
	e.SetState(payload)
	e.AccGrad(mat.NewDense[T](mat.WithBacking([]T{10, 20, 30})))

	mat.RequireMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{1, 2, 3})), e.Value())
	require.True(t, e.HasGrad())
	mat.RequireMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{10, 20, 30})), e.Grad())
	assertPayloadEqual(t, payload, e.GetOrSetState(nil).([]mat.Matrix))

	e.ReplaceValue(mat.NewDense[T](mat.WithBacking([]T{7, 8, 9})))

	mat.RequireMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{7, 8, 9})), e.Value())
	require.False(t, e.HasGrad())
	require.Nil(t, e.Grad())
	assert.Nil(t, e.GetOrSetState(nil))
}

func TestEmbedding_ScalarValue(t *testing.T) {
	type T = float32

	m := embedding.New[T](1, 1)

	e, _ := m.Embedding(0)

	e.ReplaceValue(mat.Scalar[T](42))
	assert.Equal(t, float.Interface(T(42)), e.Value().Scalar())
}

func TestEmbedding_Grad(t *testing.T) {
	type T = float32

	m := embedding.New[T](2, 3)

	e1, _ := m.Embedding(0)
	e2, _ := m.Embedding(0)

	assert.False(t, e1.HasGrad())
	assert.False(t, e2.HasGrad())

	assert.Nil(t, e1.Grad())
	assert.Nil(t, e2.Grad())

	e1.ZeroGrad() // At this point, has no effect

	assert.False(t, e1.HasGrad())
	assert.False(t, e2.HasGrad())

	assert.Nil(t, e1.Grad())
	assert.Nil(t, e2.Grad())

	e1.AccGrad(mat.NewDense[T](mat.WithBacking([]T{1, 2, 3})))

	assert.True(t, e1.HasGrad())
	assert.True(t, e2.HasGrad())

	assert.NotNil(t, e1.Grad())
	assert.NotNil(t, e2.Grad())
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{1, 2, 3})), e1.Grad())
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{1, 2, 3})), e2.Grad())
	assert.Same(t, e1.Grad(), e2.Grad())

	e1.AccGrad(mat.NewDense[T](mat.WithBacking([]T{10, 20, 30})))

	assert.True(t, e1.HasGrad())
	assert.True(t, e2.HasGrad())

	assert.NotNil(t, e1.Grad())
	assert.NotNil(t, e2.Grad())
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{11, 22, 33})), e1.Grad())
	mat.AssertMatrixEquals(t, mat.NewDense[T](mat.WithBacking([]T{11, 22, 33})), e2.Grad())
	assert.Same(t, e1.Grad(), e2.Grad())

	e1.ZeroGrad()

	assert.False(t, e1.HasGrad())
	assert.False(t, e2.HasGrad())

	assert.Nil(t, e1.Grad())
	assert.Nil(t, e2.Grad())
}

func TestEmbedding_RequiresGrad(t *testing.T) {
	t.Run("with Trainable m", func(t *testing.T) {
		m := embedding.New[float32](1, 1)

		e, _ := m.Embedding(0)
		assert.True(t, e.RequiresGrad())
	})

	t.Run("with non-Trainable m", func(t *testing.T) {
		m := embedding.New[float32](1, 3)

		e, _ := m.Embedding(0)
		e.SetRequiresGrad(false)
		assert.False(t, e.RequiresGrad())
	})
}

func TestEmbedding_Payload(t *testing.T) {
	type T = float32

	m := embedding.New[T](2, 3)

	e1, _ := m.Embedding(0)
	e2, _ := m.Embedding(0)

	assert.Nil(t, e1.GetOrSetState(nil))
	assert.Nil(t, e2.GetOrSetState(nil))

	// Set a payload for the first time
	payload := []mat.Matrix{
		mat.NewDense[T](mat.WithBacking([]T{1, 2, 3})),
		mat.NewDense[T](mat.WithBacking([]T{4, 5, 6})),
	}
	e1.SetState(payload)

	assertPayloadEqual(t, payload, e1.GetOrSetState(nil).([]mat.Matrix))
	assertPayloadEqual(t, payload, e2.GetOrSetState(nil).([]mat.Matrix))

	// Clear payload

	e1.ClearState()

	assert.Nil(t, e1.GetOrSetState(nil))
	assert.Nil(t, e2.GetOrSetState(nil))
}

func assertPayloadEqual(t *testing.T, expected, actual []mat.Matrix) {
	t.Helper()

	assert.NotNil(t, actual)
	if actual == nil {
		return
	}

	assert.Len(t, actual, len(expected))

	for i := range expected {
		assert.Equal(t, expected[i].Data(), actual[i].Data())
	}
}