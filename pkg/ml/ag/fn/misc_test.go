// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"brillion.io/spago/pkg/mat"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestAbs_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewAbs(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.1, 0.2, 0.3, 0.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-1.0, -0.5, 0.8, 0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestSafeLog_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewLog(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{-2.3025855, -1.6094379, -1.203972, -18.420680}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-10.0, 2.5, 2.6666666666, 0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestTan_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewTan(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.1003346, 0.20271, 0.3093362, 0.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-1.01006704, 0.52054567, 0.87655113, 0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestTanh_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewTanh(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.09966799, 0.19737532, 0.29131261, 0.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-0.99006629, 0.4805214, 0.73210956, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestSigmoid_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSigmoid(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.5249791, 0.54983399, 0.574442516, 0.5}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-0.24937604, 0.12375828, 0.195566649, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestHardSigmoid_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewHardSigmoid(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.52, 0.54, 0.56, 0.5}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-0.2, 0.1, 0.16, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestHardTanh_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewHardTanh(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.1, 0.2, 0.3, 0.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-1.0, 0.5, 0.8, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestRelu_Forward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, -0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewReLU(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.1, 0.0, 0.3, 0.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-1.0, 0.0, 0.8, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestNewSoftsignForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSoftsign(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.09090909, 0.16666666, 0.23076923, 0.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-0.82644628, 0.347222222, 0.473372781, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestNewCosForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewCos(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.9950041, 0.9800665, 0.9553364, 1.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{0.09983341, -0.09933466, -0.23641616, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestNewSinForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewSin(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{0.09983341, 0.19866933, 0.2955202, 0.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-0.99500416, 0.49003328, 0.7642691, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestNewExpForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewExp(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{1.10517091, 1.22140275, 1.3498588, 1.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{-1.1051709, 0.6107013, 1.07988704, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestNewNegForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, 0.0}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewNeg(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{-0.1, -0.2, -0.3, 0.0}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{1.0, -0.5, -0.8, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}

func TestNewReciprocalForward(t *testing.T) {
	x := &variable{
		value:        mat.NewVecDense([]float64{0.1, 0.2, 0.3, -0.1}),
		grad:         nil,
		requiresGrad: true,
	}
	f := NewReciprocal(x)
	y := f.Forward()

	if !floats.EqualApprox(y.Data(), []float64{10.0, 5.0, 3.33333333, -10}, 1.0e-6) {
		t.Error("The output doesn't match the expected values")
	}

	f.Backward(mat.NewVecDense([]float64{-1.0, 0.5, 0.8, 0.0}))

	if !floats.EqualApprox(x.grad.Data(), []float64{100.0, -12.5, -8.88888888, 0.0}, 1.0e-6) {
		t.Error("The x-gradients don't match the expected values")
	}
}
