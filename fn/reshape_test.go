// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestReshape_Forward(t *testing.T) {
	t.Run("float32", testReshapeForward[float32])
	t.Run("float64", testReshapeForward[float64])
}

func testReshapeForward[T float.DType](t *testing.T) {
	x := &variable{
		value: mat.NewDense(3, 4, []T{
			0.1, 0.2, 0.3, 0.0,
			0.4, 0.5, -0.6, 0.7,
			-0.5, 0.8, -0.8, -0.1,
		}),
		grad:         nil,
		requiresGrad: true,
	}

	f := NewReshape(x, 4, 3)
	assert.Equal(t, []*variable{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{
		0.1, 0.2, 0.3,
		0.0, 0.4, 0.5,
		-0.6, 0.7, -0.5,
		0.8, -0.8, -0.1,
	}, y.Data(), 1.0e-6)

	if y.Shape()[0] != 4 || y.Shape()[1] != 3 {
		t.Error("The rows and columns of the resulting matrix are not correct")
	}

	err = f.Backward(mat.NewDense(4, 3, []T{
		0.1, 0.2, 0.3,
		0.0, 0.4, 0.5,
		-0.6, 0.7, -0.5,
		0.8, -0.8, -0.1,
	}))
	assert.Nil(t, err)
	assert.InDeltaSlice(t, []T{
		0.1, 0.2, 0.3,
		0.0, 0.4, 0.5,
		-0.6, 0.7, -0.5,
		0.8, -0.8, -0.1,
	}, x.grad.Data(), 1.0e-6)

	if x.grad.Shape()[0] != 3 || x.grad.Shape()[1] != 4 {
		t.Error("The rows and columns of the resulting x-gradients matrix are not correct")
	}
}
