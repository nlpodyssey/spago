// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings_test

import (
	"bytes"
	"encoding/gob"
	"testing"

	"github.com/nlpodyssey/spago/embeddings"
	"github.com/nlpodyssey/spago/embeddings/store/memstore"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParamsMap(t *testing.T) {
	t.Run("params with gradients are traversable", func(t *testing.T) {
		type T = float32

		repo := memstore.NewRepository()
		conf := embeddings.Config{
			Size:      3,
			StoreName: "test-store",
			Trainable: true,
		}
		m := embeddings.New[T, string](conf, repo)

		// foo has grad
		foo, _ := m.Embedding("foo")
		foo.ReplaceValue(mat.NewVecDense([]T{1, 2, 3}))
		foo.AccGrad(mat.NewVecDense([]T{10, 20, 30}))

		// bar has grad too
		bar, _ := m.Embedding("bar")
		bar.ReplaceValue(mat.NewVecDense([]T{4, 5, 6}))
		bar.AccGrad(mat.NewVecDense([]T{40, 50, 60}))

		// baz has no grad
		baz, _ := m.Embedding("baz")
		baz.ReplaceValue(mat.NewVecDense([]T{7, 8, 9}))

		var visitedParamNames []string
		nn.ForEachParamStrict(m, func(p nn.Param, _ string, _ nn.ParamsType) {
			visitedParamNames = append(visitedParamNames, p.Name())
			switch p.Name() {
			case "foo":
				assert.Same(t, foo, p)
			case "bar":
				assert.Same(t, bar, p)
			default:
				t.Errorf("unexpected param %#v", p.Name())
			}
		})
		assert.Len(t, visitedParamNames, 2)
		assert.Contains(t, visitedParamNames, "foo")
		assert.Contains(t, visitedParamNames, "bar")
	})
}

func TestParamsMap_MarshalBinary(t *testing.T) {
	type T = float32

	st := embeddings.ParamsMap{
		"foo": nn.NewParam(mat.NewScalar[T](42)),
	}

	data, err := st.MarshalBinary()
	assert.NoError(t, err)
	assert.Empty(t, data)
}

func TestParamsMap_UnmarshalBinary(t *testing.T) {
	t.Run("nil", func(t *testing.T) {
		s := new(embeddings.ParamsMap)
		assert.NoError(t, s.UnmarshalBinary(nil))
		assert.Empty(t, s)
	})

	t.Run("empty slice", func(t *testing.T) {
		s := new(embeddings.ParamsMap)
		assert.NoError(t, s.UnmarshalBinary([]byte{}))
		assert.Empty(t, s)
	})

	t.Run("non empty slice", func(t *testing.T) {
		s := new(embeddings.ParamsMap)
		assert.Error(t, s.UnmarshalBinary([]byte{1}))
		assert.Empty(t, s)
	})
}

func TestGobEncoding(t *testing.T) {
	// Having already tested the marshaling methods, also testing that gob
	// works as expected is a little redundant.
	// However, since gob is the primary method used in spaGO to serialize all
	// models, it's better to be extra safe than sorry :-)

	type T = float32

	type MyStruct struct {
		Foo map[string]nn.Param
		Bar embeddings.ParamsMap
	}

	var data []byte
	{
		ms := MyStruct{
			Foo: map[string]nn.Param{
				"foo": nn.NewParam(mat.NewScalar[T](11)),
			},
			Bar: embeddings.ParamsMap{
				"bar": nn.NewParam(mat.NewScalar[T](22)),
			},
		}
		var buf bytes.Buffer
		require.NoError(t, gob.NewEncoder(&buf).Encode(ms))
		data = buf.Bytes()
	}

	var ms MyStruct
	require.NoError(t, gob.NewDecoder(bytes.NewReader(data)).Decode(&ms))

	require.NotNil(t, ms.Foo)
	require.Contains(t, ms.Foo, "foo")
	assert.Equal(t, float.Interface(T(11)), ms.Foo["foo"].Value().Scalar())

	require.Nil(t, ms.Bar)
}
