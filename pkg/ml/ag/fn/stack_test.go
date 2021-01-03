// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestStack_Forward(t *testing.T) {
	x1 := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.5}),
		grad:         nil,
		requiresGrad: true,
	}
	x2 := &variable{
		value:        mat.NewVecDense([]mat.Float{0.4, 0.5, 0.6, 0.4}),
		grad:         nil,
		requiresGrad: true,
	}
	x3 := &variable{
		value:        mat.NewVecDense([]mat.Float{0.8, 0.9, 0.7, 0.6}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewStack([]Operand{x1, x2, x3})
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.1, 0.2, 0.3, 0.5, 0.4, 0.5, 0.6, 0.4, 0.8, 0.9, 0.7, 0.6}, y.Data(), 1.0e-6)

	if y.Rows() != 3 && y.Columns() != 4 {
		t.Error("The output size doesn't match the expected values")
	}

	f.Backward(mat.NewDense(3, 4, []mat.Float{
		1.0, 2.0, 3.0, 4.0,
		4.0, 5.0, 6.0, 0.5,
		7.0, 8.0, 9.0, -0.3,
	}))

	assert.InDeltaSlice(t, []mat.Float{1.0, 2.0, 3.0, 4.0}, x1.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []mat.Float{4.0, 5.0, 6.0, 0.5}, x2.grad.Data(), 1.0e-6)
	assert.InDeltaSlice(t, []mat.Float{7.0, 8.0, 9.0, -0.3}, x3.grad.Data(), 1.0e-6)
}
