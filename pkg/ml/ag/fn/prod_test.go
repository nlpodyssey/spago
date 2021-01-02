// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestProd_Forward(t *testing.T) {
	x1 := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	x2 := &variable{
		value:        mat.NewVecDense([]float64{0.4, 0.3, 0.5, 0.7}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewProd(x1, x2)
	y := f.Forward()

	assert.InDeltaSlice(t, []float64{0.04, 0.06, 0.15, 0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []float64{-0.4, 0.15, 0.4, 0}, x1.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []float64{-0.1, 0.1, 0.24, 0}, x2.grad.Data(), 1.0e-6)
}
