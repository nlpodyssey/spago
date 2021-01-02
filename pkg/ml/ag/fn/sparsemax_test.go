// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSparseMax_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.8053, 0.4594, -0.6136, -0.9460, 1.0722}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewSparseMax(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.3597, 0.0138, 0.0000, 0.0000, 0.6265}, y.Data(), 1.0e-3)

	f.Backward(mat.NewVecDense([]mat.Float{1.0, 0.5, 0.5, 0.5, 1.0, 1.0}))

	assert.InDeltaSlice(t, []mat.Float{0.16, -0.33, 0, 0, 0.16}, x.grad.Data(), 1.0e-2)
}

func TestSparseMaxLoss_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{-0.3218, 0.7395, -0.2319, 0.2312, 0.7185}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewSparseMaxLoss(x)

	y := f.Forward()
	assert.InDeltaSlice(t, []mat.Float{-1.3009, -0.2396, -1.2110, -0.7479, -0.2606}, y.Data(), 1.0e-2)

	f.Backward(mat.NewVecDense([]mat.Float{0, 0, -1, 0, 0}))

	assert.InDeltaSlice(t, []mat.Float{0.0000, 0.5098, -1.0000, 0.0015, 0.4888}, x.grad.Data(), 1.0e-2)
}
