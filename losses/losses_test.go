// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package losses

import (
	"fmt"
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/stretchr/testify/assert"
)

func TestMSELoss(t *testing.T) {
	t.Run("float32", testMSELoss[float32])
	t.Run("float64", testMSELoss[float64])
}

func testMSELoss[T float.DType](t *testing.T) {
	x := ag.Var(mat.NewVecDense([]T{0.0, 0.1, 0.2, 0.3})).WithGrad(true)
	y := ag.Var(mat.NewVecDense([]T{0.3, 0.2, 0.1, 0.0}))
	loss := MSE(x, y, false)

	assertScalarEqualApprox(t, 0.1, loss.Value())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{-0.3, -0.1, 0.1, 0.3}, x.Grad().Data(), 1.0e-6)
}

func TestNLLLoss(t *testing.T) {
	t.Run("float32", testNLLLoss[float32])
	t.Run("float64", testNLLLoss[float64])
}

func testNLLLoss[T float.DType](t *testing.T) {
	x := ag.Var(mat.NewVecDense([]T{-0.8, 0.1, 0.693147, 1.94591})).WithGrad(true)
	y := ag.Var(mat.NewVecDense([]T{0.0, 0.0, 1.0, 0.0}))
	loss := NLL(ag.Softmax(x), y)

	fmt.Println(loss.Value().Scalar())
	assertScalarEqualApprox(t, 1.663405, loss.Value())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.042572, 0.104711, -0.810507, 0.663224}, x.Grad().Data(), 1.0e-6)
}

func TestCrossEntropyLoss(t *testing.T) {
	t.Run("float32", testCrossEntropyLoss[float32])
	t.Run("float64", testCrossEntropyLoss[float64])
}

func testCrossEntropyLoss[T float.DType](t *testing.T) {
	x := ag.Var(mat.NewVecDense([]T{-500, 0, 0.693147, 1.94591})).WithGrad(true)
	loss := CrossEntropy(x, 2)

	assertScalarEqualApprox(t, 1.609438, loss.Value())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.0, 0.1, -0.8, 0.7}, x.Grad().Data(), 1.0e-6)
}

func TestWeightedCrossEntropyLoss(t *testing.T) {
	t.Run("float32", testWeightedCrossEntropyLoss[float32])
	t.Run("float64", testWeightedCrossEntropyLoss[float64])
}

func testWeightedCrossEntropyLoss[T float.DType](t *testing.T) {
	x := ag.Var(mat.NewVecDense([]T{-500, 0, 0.693147, 1.94591})).WithGrad(true)
	w := []T{0.5, 0.5, 0.5, 0.9}
	lossFn := WeightedCrossEntropy(mat.NewVecDense(w))
	loss := lossFn(x, 2)

	assertScalarEqualApprox(t, 0.804719, loss.Value())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.0, 0.05, -0.4, 0.35}, x.Grad().Data(), 1.0e-6)
}

func TestFocalLoss(t *testing.T) {
	t.Run("float32", testFocalLoss[float32])
	t.Run("float64", testFocalLoss[float64])
}

func testFocalLoss[T float.DType](t *testing.T) {
	x := ag.Var(mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.4})).WithGrad(true)
	loss := FocalLoss(x, 2, 2.0)

	assertScalarEqualApprox(t, 0.73282546, loss.Value())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.22751944, 0.25144786, -0.78608638, 0.3071191}, x.Grad().Data(), 1.0e-6)
}

func TestWeightedFocalLoss(t *testing.T) {
	t.Run("float32", testWeightedFocalLoss[float32])
	t.Run("float64", testWeightedFocalLoss[float64])
}

func testWeightedFocalLoss[T float.DType](t *testing.T) {
	x := ag.Var(mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.4})).WithGrad(true)
	w := []T{0.5, 0.5, 0.5, 0.9}
	lossFn := WeightedFocalLoss(mat.NewVecDense(w))
	loss := lossFn(x, 2, 2.0)

	assertScalarEqualApprox(t, 0.36641273, loss.Value())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.11375972, 0.12572393, -0.39304319, 0.15355955}, x.Grad().Data(), 1.0e-6)
}

func TestZeroOneQuantization(t *testing.T) {
	t.Run("float32", testZeroOneQuantization[float32])
	t.Run("float64", testZeroOneQuantization[float64])
}

func testZeroOneQuantization[T float.DType](t *testing.T) {
	x := ag.Var(mat.NewVecDense([]T{0.1, 0.2, 1.0, 0.4, -0.8, 0.3})).WithGrad(true)
	loss := ZeroOneQuantization(x)

	assertScalarEqualApprox(t, 2.209, loss.Value())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.144, 0.192, 0.0, 0.096, -7.488, 0.168}, x.Grad().Data(), 1.0e-6)
}

func TestNorm2Quantization(t *testing.T) {
	t.Run("float32", testNorm2Quantization[float32])
	t.Run("float64", testNorm2Quantization[float64])
}

func testNorm2Quantization[T float.DType](t *testing.T) {
	x := ag.Var(mat.NewVecDense([]T{0.1, 0.2, 1.0, 0.4, -0.8, 0.3})).WithGrad(true)
	loss := Norm2Quantization(x)

	assertScalarEqualApprox(t, 0.8836, loss.Value())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.376, 0.752, 3.76, 1.504, -3.008, 1.128}, x.Grad().Data(), 1.0e-6)
}

func TestOneHotQuantization(t *testing.T) {
	t.Run("float32", testOneHotQuantization[float32])
	t.Run("float64", testOneHotQuantization[float64])
}

func testOneHotQuantization[T float.DType](t *testing.T) {
	x := ag.Var(mat.NewVecDense([]T{0.1, 0.2, 1.0, 0.4, -0.8, 0.3})).WithGrad(true)
	loss := OneHotQuantization(x, 0.1)

	assertScalarEqualApprox(t, 0.30926, loss.Value())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.052, 0.0944, 0.376, 0.16, -1.0496, 0.1296}, x.Grad().Data(), 1.0e-6)
}

func TestMSESeqLoss(t *testing.T) {
	t.Run("float32", testMSESeqLoss[float32])
	t.Run("float64", testMSESeqLoss[float64])
}

func testMSESeqLoss[T float.DType](t *testing.T) {
	x1 := ag.Var(mat.NewVecDense([]T{0.0, 0.1, 0.2, 0.3})).WithGrad(true)
	y1 := ag.Var(mat.NewVecDense([]T{0.3, 0.2, 0.1, 0.0}))
	x2 := ag.Var(mat.NewVecDense([]T{0.0, 0.1, 0.2, 0.3})).WithGrad(true)
	y2 := ag.Var(mat.NewVecDense([]T{0.3, 0.2, 0.1, 0.0}))
	loss := MSESeq([]ag.Node{x1, x2}, []ag.Node{y1, y2}, true)

	assertScalarEqualApprox(t, 0.1, loss.Value())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{-0.15, -0.05, 0.05, 0.15}, x1.Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{-0.15, -0.05, 0.05, 0.15}, x2.Grad().Data(), 1.0e-6)
}

func assertScalarEqualApprox[T float.DType](t *testing.T, expected T, actual mat.Matrix) {
	t.Helper()
	v := float.ValueOf[T](actual.Scalar())
	assert.InDelta(t, expected, v, 1.0e-06)
}
