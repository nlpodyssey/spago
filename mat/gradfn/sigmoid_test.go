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

func TestSigmoid_Forward(t *testing.T) {
	t.Run("float32", testSigmoidForward[float32])
	t.Run("float64", testSigmoidForward[float64])
}

func testSigmoidForward[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 0.3, 0.0}), mat.WithGrad(true))
	f := NewSigmoid(x)
	y, err := f.Forward()
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{0.5249791, 0.54983399, 0.574442516, 0.5}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewDense[T](mat.WithBacking([]T{-1.0, 0.5, 0.8, 0.0})))
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{-0.24937604, 0.12375828, 0.195566649, 0.0}, x.Grad().Data(), 1.0e-6)
}
