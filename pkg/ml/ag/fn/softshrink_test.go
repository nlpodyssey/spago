// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSoftShrink_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, -0.2, 0.3, 0.0, 0.6, -0.6}),
		grad:         nil,
		requiresGrad: true,
	}
	lambda := &variable{
		value:        mat.NewScalar(0.2),
		grad:         nil,
		requiresGrad: false,
	}

	f := NewSoftShrink(x, lambda)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.0, 0.0, 0.1, 0, 0.4, -0.4}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0, 1.0, 2.0}))

	assert.InDeltaSlice(t, []mat.Float{0.0, 0.0, 0.8, 0.0, 1.0, 2.0}, x.grad.Data(), 1.0e-6)
}
