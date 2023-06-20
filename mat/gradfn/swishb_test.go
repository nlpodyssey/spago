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

func TestSwishBForward(t *testing.T) {
	t.Run("float32", testSwishBForward[float32])
	t.Run("float64", testSwishBForward[float64])
}

func testSwishBForward[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{0.1, -0.2, 0.3, 0.0}), mat.WithGrad(true))
	beta := mat.NewDense[T](mat.WithBacking([]T{2.0}), mat.WithGrad(true))
	f := NewSwishB(x, beta)
	assert.Equal(t, []mat.Tensor{x, beta}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.0549833997, -0.080262468, 0.1936968919, 0.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{-1.0, 0.5, 0.8, 0.0})))
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{-0.5993373119, 0.1526040208, 0.6263414804, 0.0}, x.Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{0.0188025145}, beta.Grad().Data(), 1.0e-6)
}
