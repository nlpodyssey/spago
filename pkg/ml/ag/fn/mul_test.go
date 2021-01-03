// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMul_ForwardMatrixMatrix(t *testing.T) {
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
		value: mat.NewDense(4, 3, []mat.Float{
			0.2, 0.7, 0.5,
			0.0, 0.4, 0.5,
			-0.8, 0.7, -0.3,
			0.2, -0.0, -0.9,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewMul(x1, x2)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{
		-0.22, 0.36, 0.06,
		0.7, 0.06, 0.0,
		0.52, -0.59, 0.48,
	}, y.Data(), 1.0e-6)

	f.Backward(mat.NewDense(3, 3, []mat.Float{
		0.2, 0.7, 0.5,
		0.0, 0.4, 0.5,
		-0.6, 0.7, -0.5,
	}))

	assert.InDeltaSlice(t, []mat.Float{
		0.78, 0.53, 0.18, -0.41,
		0.53, 0.41, 0.13, -0.45,
		0.12, 0.03, 1.12, 0.33,
	}, x1.grad.Data(), 1.0e-6)

	assert.InDeltaSlice(t, []mat.Float{
		0.32, -0.12, 0.5,
		-0.44, 0.9, -0.05,
		0.54, -0.59, 0.25,
		0.06, 0.21, 0.4,
	}, x2.grad.Data(), 1.0e-2)
}

func TestMul_ForwardMatrixVector(t *testing.T) {
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
		value:        mat.NewVecDense([]mat.Float{-0.8, -0.9, -0.9, 1.0}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewMul(x1, x2)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{-0.53, 0.47, 0.3}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{0.2, -0.6, 0.8}))

	assert.InDeltaSlice(t, []mat.Float{
		-0.16, -0.18, -0.18, 0.2,
		0.48, 0.54, 0.54, -0.6,
		-0.64, -0.72, -0.72, 0.8,
	}, x1.grad.Data(), 1.0e-6)

	if x1.grad.Rows() != 3 || x1.grad.Columns() != 4 {
		t.Error("The rows and columns of the resulting x1-gradients are not correct")
	}

	assert.InDeltaSlice(t, []mat.Float{-0.62, 0.38, -0.22, -0.5}, x2.grad.Data(), 1.0e-6)
}
