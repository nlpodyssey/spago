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

func TestAbs_Forward(t *testing.T) {
	t.Run("float32", testAbsForward[float32])
	t.Run("float64", testAbsForward[float64])
}

func testAbsForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewAbs(x)
	assert.Equal(t, []*variable{x}, f.Operands())

	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.1, 0.2, 0.3, 0.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-1.0, -0.5, 0.8, 0}, x.grad.Data(), 1.0e-6)
}

func TestSafeLog_Forward(t *testing.T) {
	t.Run("float32", testSafeLogForward[float32])
	t.Run("float64", testSafeLogForward[float64])
}

func testSafeLogForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewLog(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-2.3025855, -1.6094379, -1.203972, mat.Inf[T](-1)}, y.Data(), 1.0e-5)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-10.0, 2.5, 2.6666666666, 0}, x.grad.Data(), 1.0e-6)
}

func TestTan_Forward(t *testing.T) {
	t.Run("float32", testTanForward[float32])
	t.Run("float64", testTanForward[float64])
}

func testTanForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewTan(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.1003346, 0.20271, 0.3093362, 0.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-1.01006704, 0.52054567, 0.87655113, 0}, x.grad.Data(), 1.0e-6)
}

func TestTanh_Forward(t *testing.T) {
	t.Run("float32", testTanhForward[float32])
	t.Run("float64", testTanhForward[float64])
}

func testTanhForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewTanh(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.09966799, 0.19737532, 0.29131261, 0.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-0.99006629, 0.4805214, 0.73210956, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestHardSigmoid_Forward(t *testing.T) {
	t.Run("float32", testHardSigmoidForward[float32])
	t.Run("float64", testHardSigmoidForward[float64])
}

func testHardSigmoidForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewHardSigmoid(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.52, 0.54, 0.56, 0.5}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-0.2, 0.1, 0.16, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestHardTanh_Forward(t *testing.T) {
	t.Run("float32", testHardTanhForward[float32])
	t.Run("float64", testHardTanhForward[float64])
}

func testHardTanhForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewHardTanh(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.1, 0.2, 0.3, 0.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-1.0, 0.5, 0.8, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestRelu_Forward(t *testing.T) {
	t.Run("float32", testReluForward[float32])
	t.Run("float64", testReluForward[float64])
}

func testReluForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewReLU(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.1, 0.0, 0.3, 0.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.NoError(t, err)

	assert.InDeltaSlice(t, []T{-1.0, 0.0, 0.8, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewSoftsignForward(t *testing.T) {
	t.Run("float32", testNewSoftsignForward[float32])
	t.Run("float64", testNewSoftsignForward[float64])
}

func testNewSoftsignForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSoftsign(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.09090909, 0.16666666, 0.23076923, 0.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-0.82644628, 0.347222222, 0.473372781, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewCosForward(t *testing.T) {
	t.Run("float32", testNewCosForward[float32])
	t.Run("float64", testNewCosForward[float64])
}

func testNewCosForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewCos(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.9950041, 0.9800665, 0.9553364, 1.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.09983341, -0.09933466, -0.23641616, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewSinForward(t *testing.T) {
	t.Run("float32", testNewSinForward[float32])
	t.Run("float64", testNewSinForward[float64])
}

func testNewSinForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSin(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.09983341, 0.19866933, 0.2955202, 0.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-0.99500416, 0.49003328, 0.7642691, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewExpForward(t *testing.T) {
	t.Run("float32", testNewExpForward[float32])
	t.Run("float64", testNewExpForward[float64])
}

func testNewExpForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewExp(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{1.10517091, 1.22140275, 1.3498588, 1.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-1.1051709, 0.6107013, 1.07988704, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewNegForward(t *testing.T) {
	t.Run("float32", testNewNegForward[float32])
	t.Run("float64", testNewNegForward[float64])
}

func testNewNegForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewNeg(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-0.1, -0.2, -0.3, 0.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.NoError(t, err)

	assert.InDeltaSlice(t, []T{1.0, -0.5, -0.8, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewReciprocalForward(t *testing.T) {
	t.Run("float32", testNewReciprocalForward[float32])
	t.Run("float64", testNewReciprocalForward[float64])
}

func testNewReciprocalForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, -0.1}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewReciprocal(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{10.0, 5.0, 3.33333333, -10}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{100.0, -12.5, -8.88888888, 0.0}, x.grad.Data(), 1.0e-5)
}

func TestNewMishForward(t *testing.T) {
	t.Run("float32", testNewMishForward[float32])
	t.Run("float64", testNewMishForward[float64])
}

func testNewMishForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.1, 0.2, 0.3, -0.1}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewMish(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.0631794175, 0.1325990019, 0.2080013723, -0.0567885752}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{-1.0, 0.5, 0.8, 0.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{-0.6633368208, 0.3623122702, 0.6262618396, 0.0}, x.grad.Data(), 1.0e-6)
}

func TestNewGELUForward(t *testing.T) {
	t.Run("float32", testNewGELUForward[float32])
	t.Run("float64", testNewGELUForward[float64])
}

func testNewGELUForward[T float.DType](t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]T{0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewGELU(x)
	y, err := f.Forward()
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.0, 0.053983, 0.00504, -0.046017, -0.00496, 0.841192, 10.0, -0.158808, 0.0}, y.Data(), 1.0e-6)

	err = f.Backward(mat.NewVecDense([]T{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}))
	assert.Nil(t, err)

	assert.InDeltaSlice(t, []T{0.5, 0.579522, 0.507979, 0.420478, 0.492021, 1.082964, 1.0, -0.082964, 0.0}, x.grad.Data(), 1.0e-6)
}
