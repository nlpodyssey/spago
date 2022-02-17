// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestUtils(t *testing.T) {
	t.Run("float32", testUtils[float64])
	t.Run("float64", testUtils[float64])
}

func testUtils[T mat.DType](t *testing.T) {
	t.Run("test `Map2`", func(t *testing.T) {
		g := NewGraph[T]()
		ys := Map2(Add[T],
			[]Node[T]{g.NewScalar(1), g.NewScalar(2), g.NewScalar(3)},
			[]Node[T]{g.NewScalar(4), g.NewScalar(5), g.NewScalar(6)},
		)
		assert.Equal(t, 3, len(ys))
		assert.Equal(t, T(5), ys[0].ScalarValue())
		assert.Equal(t, T(7), ys[1].ScalarValue())
		assert.Equal(t, T(9), ys[2].ScalarValue())
	})

	t.Run("test `Pad`", func(t *testing.T) {
		g := NewGraph[T]()
		newEl := func(_ int) Node[T] {
			return g.NewScalar(0)
		}
		ys := Pad([]Node[T]{g.NewScalar(1), g.NewScalar(2), g.NewScalar(3)}, 5, newEl)
		assert.Equal(t, 5, len(ys))
		assert.Equal(t, T(1), ys[0].ScalarValue())
		assert.Equal(t, T(2), ys[1].ScalarValue())
		assert.Equal(t, T(3), ys[2].ScalarValue())
		assert.Equal(t, T(0), ys[3].ScalarValue())
		assert.Equal(t, T(0), ys[4].ScalarValue())
	})

	t.Run("test `Pad` with no need to pad", func(t *testing.T) {
		g := NewGraph[T]()
		newEl := func(_ int) Node[T] {
			return g.NewScalar(0)
		}
		ys := Pad([]Node[T]{g.NewScalar(1), g.NewScalar(2), g.NewScalar(3)}, 3, newEl)
		assert.Equal(t, 3, len(ys))
		assert.Equal(t, T(1), ys[0].ScalarValue())
		assert.Equal(t, T(2), ys[1].ScalarValue())
		assert.Equal(t, T(3), ys[2].ScalarValue())
	})
}
