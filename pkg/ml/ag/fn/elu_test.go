// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestELUForward(t *testing.T) {
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
	f := NewELU(x, alpha)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.1, -0.36253849, 0.3, 0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-1.0, 0.8187307, 0.8, 0.0}, x.grad.Data(), 1.0e-6)
}
