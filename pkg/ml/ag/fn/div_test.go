// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDiv_Forward(t *testing.T) {
	x1 := &variable[mat.Float]{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	x2 := &variable[mat.Float]{
		value:        mat.NewVecDense([]mat.Float{0.4, 0.3, 0.5, 0.7}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewDiv[mat.Float](x1, x2)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.25, 0.6666666666, 0.6, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-2.5, 1.6666666666666, 1.6, 0}, x1.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []mat.Float{0.625, -1.11111111111111, -0.96, 0}, x2.grad.Data(), 1.0e-6)
}
