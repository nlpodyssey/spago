// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestView_Forward(t *testing.T) {
	x := &variable{
		value: mat.NewDense(3, 4, []mat.Float{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewView(x, 1, 1, 2, 2)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{
		0.5, -0.6,
		0.8, -0.8,
	}, y.Data(), 1.0e-6)

	if y.Rows() != 2 || y.Columns() != 2 {
		t.Error("The rows and columns of the resulting matrix are not correct")
	}

	f.Backward(mat.NewDense(2, 2, []mat.Float{
		0.1, 0.2,
		-0.8, -0.1,
	}))

	assert.InDeltaSlice(t, []mat.Float{
		0.0, 0.0, 0.0, 0.0,
		0.0, 0.1, 0.2, 0.0,
		0.0, -0.8, -0.1, 0.0,
	}, x.grad.Data(), 1.0e-6)
}
