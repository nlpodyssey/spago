// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSwishBForward(t *testing.T) {
	x := &variable[mat.Float]{
		value:        mat.NewVecDense([]mat.Float{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	beta := &variable[mat.Float]{
		value:        mat.NewScalar[mat.Float](2.0),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSwishB[mat.Float](x, beta)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.0549833997, -0.080262468, 0.1936968919, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-0.5993373119, 0.1526040208, 0.6263414804, 0.0}, x.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []mat.Float{0.0188025145}, beta.grad.Data(), 1.0e-6)
}
