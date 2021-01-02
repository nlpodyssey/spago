// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDot_Forward(t *testing.T) {

	x1 := &variable{
		value: mat.NewDense(3, 4, []mat.Float{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	x2 := &variable{
		value: mat.NewDense(3, 4, []mat.Float{
			0.1, 0.8, 0.3, 0.1,
			0.1, -0.5, -0.9, 0.2,
			-0.2, 0.3, -0.4, -0.5,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewDot(x1, x2)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{1.44}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{0.5}))

	assert.InDeltaSlice(t, []mat.Float{
		0.05, 0.4, 0.15, 0.05,
		0.05, -0.25, -0.45, 0.1,
		-0.1, 0.15, -0.2, -0.25,
	}, x1.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []mat.Float{
		0.05, 0.1, 0.15, 0.0,
		0.2, 0.25, -0.3, 0.35,
		-0.25, 0.4, -0.4, -0.05,
	}, x2.grad.Data(), 1.0e-6)
}
