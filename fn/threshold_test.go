// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestThresholdForward(t *testing.T) {
	t.Run("float32", testThresholdForward[float32])
	t.Run("float64", testThresholdForward[float64])
}

func testThresholdForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewDense[T](mat.WithBacking([]T{0.1, -0.2, 3.3, 0.0})),
		grad:         nil,
		requiresGrad: true,
	}
	ts := &variable{
		value:        mat.Scalar[T](2.0),
		grad:         nil,
		requiresGrad: false,
	}
	k := &variable{
		value:        mat.Scalar[T](1.6),
		grad:         nil,
		requiresGrad: false,
	}

	f := NewThreshold(x, ts, k)
	assert.Equal(t, []*variable{x, ts, k}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{1.6, 1.6, 3.3, 1.6}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{-1.0, 0.5, 0.8, 0.0})))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.0, 0.0, 0.8, 0}, x.grad.Data(), 1.0e-6)
}
