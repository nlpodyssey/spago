// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradfn

import (
	"github.com/stretchr/testify/assert"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

func TestReduceMean_Forward(t *testing.T) {
	t.Run("float32", testReduceMeanForward[float32])
	t.Run("float64", testReduceMeanForward[float64])
}

func testReduceMeanForward[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 0.3, 0.0}), mat.WithGrad(true))
	f := NewReduceMean(x)
	assert.Equal(t, []mat.Tensor{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.15}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{0.5})))
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.125, 0.125, 0.125, 0.125}, x.Grad().Data(), 1.0e-6)
}
