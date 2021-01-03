// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestAbs_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewAbs(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.1, 0.2, 0.3, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-1.0, -0.5, 0.8, 0}, x.grad.Data(), 1.0e-6)
}

func TestSafeLog_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewLog(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{-2.3025855, -1.6094379, -1.203972, -18.420680}, y.Data(), 1.0e-5)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-10.0, 2.5, 2.6666666666, 0}, x.grad.Data(), 1.0e-6)
}

func TestTan_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewTan(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.1003346, 0.20271, 0.3093362, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-1.01006704, 0.52054567, 0.87655113, 0}, x.grad.Data(), 1.0e-6)
}

func TestTanh_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewTanh(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.09966799, 0.19737532, 0.29131261, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-0.99006629, 0.4805214, 0.73210956, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestSigmoid_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSigmoid(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.5249791, 0.54983399, 0.574442516, 0.5}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-0.24937604, 0.12375828, 0.195566649, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestHardSigmoid_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewHardSigmoid(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.52, 0.54, 0.56, 0.5}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-0.2, 0.1, 0.16, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestHardTanh_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewHardTanh(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.1, 0.2, 0.3, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-1.0, 0.5, 0.8, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestRelu_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewReLU(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.1, 0.0, 0.3, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-1.0, 0.0, 0.8, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewSoftsignForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSoftsign(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.09090909, 0.16666666, 0.23076923, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-0.82644628, 0.347222222, 0.473372781, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewCosForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewCos(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.9950041, 0.9800665, 0.9553364, 1.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{0.09983341, -0.09933466, -0.23641616, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewSinForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSin(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.09983341, 0.19866933, 0.2955202, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-0.99500416, 0.49003328, 0.7642691, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewExpForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewExp(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{1.10517091, 1.22140275, 1.3498588, 1.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-1.1051709, 0.6107013, 1.07988704, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewNegForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewNeg(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{-0.1, -0.2, -0.3, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{1.0, -0.5, -0.8, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewReciprocalForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, -0.1}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewReciprocal(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{10.0, 5.0, 3.33333333, -10}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{100.0, -12.5, -8.88888888, 0.0}, x.grad.Data(), 1.0e-5)
}

func TestNewMishForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.1, 0.2, 0.3, -0.1}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewMish(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.0631794175, 0.1325990019, 0.2080013723, -0.0567885752}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0}))

	assert.InDeltaSlice(t, []mat.Float{-0.6633368208, 0.3623122702, 0.6262618396, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewGELUForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]mat.Float{0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewGELU(x)
	y := f.Forward()

	assert.InDeltaSlice(t, []mat.Float{0.0, 0.053983, 0.00504, -0.046017, -0.00496, 0.841192, 10.0, -0.158808, 0.0}, y.Data(), 1.0e-6)

	f.Backward(mat.NewVecDense([]mat.Float{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))

	assert.InDeltaSlice(t, []mat.Float{0.5, 0.579522, 0.507979, 0.420478, 0.492021, 1.082964, 1.0, -0.082964, 0.0}, x.grad.Data(), 1.0e-6)
}
