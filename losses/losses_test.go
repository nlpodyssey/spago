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
	x := mat.NewDense[T](mat.WithBacking([]T{0.0, 0.1, 0.2, 0.3}), mat.WithGrad(true))
	y := mat.NewDense[T](mat.WithBacking([]T{0.3, 0.2, 0.1, 0.0}))
	loss := MSE(x, y, false)

	assertScalarEqualApprox(t, 0.1, loss.Value().(mat.Matrix))

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{-0.3, -0.1, 0.1, 0.3}, x.Grad().Data(), 1.0e-6)
}

func TestNLLLoss(t *testing.T) {
	t.Run("float32", testNLLLoss[float32])
	t.Run("float64", testNLLLoss[float64])
}

func testNLLLoss[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{-0.8, 0.1, 0.693147, 1.94591}), mat.WithGrad(true))
	y := mat.NewDense[T](mat.WithBacking([]T{0.0, 0.0, 1.0, 0.0}))
	loss := NLL(ag.Softmax(x), y)

	fmt.Println(loss.Value().Item())
	assertScalarEqualApprox(t, 1.663405, loss.Value().(mat.Matrix))

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.042572, 0.104711, -0.810507, 0.663224}, x.Grad().Data(), 1.0e-6)
}

func TestCrossEntropyLoss(t *testing.T) {
	t.Run("float32", testCrossEntropyLoss[float32])
	t.Run("float64", testCrossEntropyLoss[float64])
}

func testCrossEntropyLoss[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{-500, 0, 0.693147, 1.94591}), mat.WithGrad(true))
	loss := CrossEntropy(x, 2)

	assertScalarEqualApprox(t, 1.609438, loss.Value().(mat.Matrix))

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.0, 0.1, -0.8, 0.7}, x.Grad().Data(), 1.0e-6)
}

func TestWeightedCrossEntropyLoss(t *testing.T) {
	t.Run("float32", testWeightedCrossEntropyLoss[float32])
	t.Run("float64", testWeightedCrossEntropyLoss[float64])
}

func testWeightedCrossEntropyLoss[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{-500, 0, 0.693147, 1.94591}), mat.WithGrad(true))
	w := []T{0.5, 0.5, 0.5, 0.9}
	lossFn := WeightedCrossEntropy(mat.NewDense[T](mat.WithBacking(w)))
	loss := lossFn(x, 2)

	assertScalarEqualApprox(t, 0.804719, loss.Value().(mat.Matrix))

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.0, 0.05, -0.4, 0.35}, x.Grad().Data(), 1.0e-6)
}

func TestFocalLoss(t *testing.T) {
	t.Run("float32", testFocalLoss[float32])
	t.Run("float64", testFocalLoss[float64])
}

func testFocalLoss[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 0.3, 0.4}), mat.WithGrad(true))
	loss := FocalLoss(x, 2, 2.0)

	assertScalarEqualApprox(t, 0.73282546, loss.Value().(mat.Matrix))

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.22751944, 0.25144786, -0.78608638, 0.3071191}, x.Grad().Data(), 1.0e-6)
}

func TestWeightedFocalLoss(t *testing.T) {
	t.Run("float32", testWeightedFocalLoss[float32])
	t.Run("float64", testWeightedFocalLoss[float64])
}

func testWeightedFocalLoss[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 0.3, 0.4}), mat.WithGrad(true))
	w := []T{0.5, 0.5, 0.5, 0.9}
	lossFn := WeightedFocalLoss(mat.NewDense[T](mat.WithBacking(w)))
	loss := lossFn(x, 2, 2.0)

	assertScalarEqualApprox(t, 0.36641273, loss.Value().(mat.Matrix))

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.11375972, 0.12572393, -0.39304319, 0.15355955}, x.Grad().Data(), 1.0e-6)
}

func TestZeroOneQuantization(t *testing.T) {
	t.Run("float32", testZeroOneQuantization[float32])
	t.Run("float64", testZeroOneQuantization[float64])
}

func testZeroOneQuantization[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), mat.WithGrad(true))
	loss := ZeroOneQuantization(x)

	assertScalarEqualApprox(t, 2.209, loss.Value().(mat.Matrix))

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.144, 0.192, 0.0, 0.096, -7.488, 0.168}, x.Grad().Data(), 1.0e-6)
}

func TestNorm2Quantization(t *testing.T) {
	t.Run("float32", testNorm2Quantization[float32])
	t.Run("float64", testNorm2Quantization[float64])
}

func testNorm2Quantization[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), mat.WithGrad(true))
	loss := Norm2Quantization(x)

	assertScalarEqualApprox(t, 0.8836, loss.Value().(mat.Matrix))

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.376, 0.752, 3.76, 1.504, -3.008, 1.128}, x.Grad().Data(), 1.0e-6)
}

func TestOneHotQuantization(t *testing.T) {
	t.Run("float32", testOneHotQuantization[float32])
	t.Run("float64", testOneHotQuantization[float64])
}

func testOneHotQuantization[T float.DType](t *testing.T) {
	x := mat.NewDense[T](mat.WithBacking([]T{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), mat.WithGrad(true))
	loss := OneHotQuantization(x, 0.1)

	assertScalarEqualApprox(t, 0.30926, loss.Value().(mat.Matrix))

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.052, 0.0944, 0.376, 0.16, -1.0496, 0.1296}, x.Grad().Data(), 1.0e-6)
}

func TestMSESeqLoss(t *testing.T) {
	t.Run("float32", testMSESeqLoss[float32])
	t.Run("float64", testMSESeqLoss[float64])
}

func testMSESeqLoss[T float.DType](t *testing.T) {
	x1 := mat.NewDense[T](mat.WithBacking([]T{0.0, 0.1, 0.2, 0.3}), mat.WithGrad(true))
	y1 := mat.NewDense[T](mat.WithBacking([]T{0.3, 0.2, 0.1, 0.0}))
	x2 := mat.NewDense[T](mat.WithBacking([]T{0.0, 0.1, 0.2, 0.3}), mat.WithGrad(true))
	y2 := mat.NewDense[T](mat.WithBacking([]T{0.3, 0.2, 0.1, 0.0}), mat.WithGrad(true))
	loss := MSESeq([]mat.Tensor{x1, x2}, []mat.Tensor{y1, y2}, true)

	assertScalarEqualApprox(t, 0.1, loss.Value().(mat.Matrix))

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{-0.15, -0.05, 0.05, 0.15}, x1.Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{-0.15, -0.05, 0.05, 0.15}, x2.Grad().Data(), 1.0e-6)
}

func assertScalarEqualApprox[T float.DType](t *testing.T, expected T, actual mat.Matrix) {
	t.Helper()
	v := float.ValueOf[T](actual.Item())
	assert.InDelta(t, expected, v, 1.0e-06)
}

func TestHuberLoss(t *testing.T) {
	t.Run("float32", func(t *testing.T) { testHuberLoss[float32](t, 1.0e-6) })
	t.Run("float64", func(t *testing.T) { testHuberLoss[float64](t, 1.0e-12) })
}

func testHuberLoss[T float.DType](t *testing.T, tol T) {
	// 1) Setup input, target, delta
	x := mat.NewDense[T](mat.WithBacking([]T{0.0, 2.5, 4.0}), mat.WithGrad(true))
	y := mat.NewDense[T](mat.WithBacking([]T{0.0, 1.0, 2.0}))
	delta := 1.0

	// 2) Compute Huber loss with reduceMean = false
	loss := Huber(x, y, delta, false)

	// Sum of the "Huber" terms across 3 elements => ~2.5
	assert.InDelta(t, 2.5, loss.Value().Item().F64(), float64(tol))

	// 3) Backward
	ag.Backward(loss)

	// For this example:
	// d = [0, 1.5, 2.0]; gradient = sign(d)*delta if |d|>delta else d
	// => [0, 1.0, 1.0]
	assert.InDeltaSlice(t, []T{0.0, 1.0, 1.0}, x.Grad().Data(), float64(tol))

	// 4) Test again with reduceMean = true
	x2 := mat.NewDense[T](mat.WithBacking([]T{0.0, 2.5, 4.0}), mat.WithGrad(true))
	y2 := mat.NewDense[T](mat.WithBacking([]T{0.0, 1.0, 2.0}))
	loss2 := Huber(x2, y2, delta, true)

	// The total is 2.5 for 3 elements => 2.5 / 3 = ~0.8333
	assert.InDelta(t, 0.8333333333333333, loss2.Value().Item().F64(), float64(tol))

	ag.Backward(loss2)
	// The gradient is the same shape but divided by 3 => [0, ~0.3333, ~0.3333]
	assert.InDeltaSlice(t, []T{0.0, 0.3333333333333333, 0.3333333333333333}, x2.Grad().Data(), float64(tol))
}
