// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSoftPlusForward(t *testing.T) {
	x := &variable[mat.Float]{
		value:        mat.NewVecDense([]mat.Float{0.1, -0.2, 20.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	beta := &variable[mat.Float]{
		value:        mat.NewScalar[mat.Float](2.0),
		grad:         nil,
		requiresGrad: false,
	}
	threshold := &variable[mat.Float]{
		value:        mat.NewScalar[mat.Float](20.0),
		grad:         nil,
		requiresGrad: false,
	}

	f := NewSoftPlus[mat.Float](x, beta, threshold)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.399069434, 0.25650762, 20.3, 0.346573590}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-0.5498339, 0.20065616, 0.8, 0}, x.grad.Data(), 1.0e-6)
}
