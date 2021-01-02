// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestPow_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewPow(x, 3.0)
	y := f.Forward()

	assert.InDeltaSlice(t, []float64{0.001, 0.008, 0.027, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []float64{-0.03, 0.06, 0.216, 0}, x.grad.Data(), 1.0e-6)
}
