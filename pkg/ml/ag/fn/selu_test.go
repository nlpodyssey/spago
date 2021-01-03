// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSELUForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	alpha := &variable{
		value:        mat.NewScalar(2.0),
		grad:         nil,
		requiresGrad: false,
	}
	scale := &variable{
		value:        mat.NewScalar(1.6),
		grad:         nil,
		requiresGrad: false,
	}

	f := NewSELU(x, alpha, scale)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.16, -0.58006159, 0.48, 0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-1.6, 1.3099692, 1.28, 0}, x.grad.Data(), 1.0e-6)
}
