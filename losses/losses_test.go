// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package losses

import (
	"fmt"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMSELoss(t *testing.T) {
	t.Run("float32", testMSELoss[float32])
	t.Run("float64", testMSELoss[float64])
}

func testMSELoss[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()
	x := g.NewVariable(mat.NewVecDense([]T{0.0, 0.1, 0.2, 0.3}), true)
	y := g.NewVariable(mat.NewVecDense([]T{0.3, 0.2, 0.1, 0.0}), false)
	loss := MSE(x, y, false)

	assertEqualApprox(t, 0.1, loss.Value().Scalar())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{-0.3, -0.1, 0.1, 0.3}, x.Grad().Data(), 1.0e-6)
}

func TestNLLLoss(t *testing.T) {
	t.Run("float32", testNLLLoss[float32])
	t.Run("float64", testNLLLoss[float64])
}

func testNLLLoss[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()
	x := g.NewVariable(mat.NewVecDense([]T{-0.8, 0.1, 0.693147, 1.94591}), true)
	y := g.NewVariable(mat.NewVecDense([]T{0.0, 0.0, 1.0, 0.0}), false)
	loss := NLL(ag.Softmax(x), y)

	fmt.Println(loss.Value().Scalar())
	assertEqualApprox(t, 1.663405, loss.Value().Scalar())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.042572, 0.104711, -0.810507, 0.663224}, x.Grad().Data(), 1.0e-6)
}

func TestCrossEntropyLoss(t *testing.T) {
	t.Run("float32", testCrossEntropyLoss[float32])
	t.Run("float64", testCrossEntropyLoss[float64])
}

func testCrossEntropyLoss[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()
	x := g.NewVariable(mat.NewVecDense([]T{-500, 0, 0.693147, 1.94591}), true)
	loss := CrossEntropy(x, 2)

	assertEqualApprox(t, 1.609438, loss.Value().Scalar())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.0, 0.1, -0.8, 0.7}, x.Grad().Data(), 1.0e-6)
}

func TestWeightedCrossEntropyLoss(t *testing.T) {
	t.Run("float32", testWeightedCrossEntropyLoss[float32])
	t.Run("float64", testWeightedCrossEntropyLoss[float64])
}

func testWeightedCrossEntropyLoss[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()
	x := g.NewVariable(mat.NewVecDense([]T{-500, 0, 0.693147, 1.94591}), true)
	w := []T{0.5, 0.5, 0.5, 0.9}
	lossFn := WeightedCrossEntropy(w)
	loss := lossFn(x, 2)

	assertEqualApprox(t, 0.804719, loss.Value().Scalar())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.0, 0.05, -0.4, 0.35}, x.Grad().Data(), 1.0e-6)
}

func TestFocalLoss(t *testing.T) {
	t.Run("float32", testFocalLoss[float32])
	t.Run("float64", testFocalLoss[float64])
}

func testFocalLoss[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()
	x := g.NewVariable(mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.4}), true)
	loss := FocalLoss(x, 2, 2.0)

	assertEqualApprox(t, 0.73282546, loss.Value().Scalar())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.22751944, 0.25144786, -0.78608638, 0.3071191}, x.Grad().Data(), 1.0e-6)
}

func TestWeightedFocalLoss(t *testing.T) {
	t.Run("float32", testWeightedFocalLoss[float32])
	t.Run("float64", testWeightedFocalLoss[float64])
}

func testWeightedFocalLoss[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()
	x := g.NewVariable(mat.NewVecDense([]T{0.1, 0.2, 0.3, 0.4}), true)
	w := []T{0.5, 0.5, 0.5, 0.9}
	lossFn := WeightedFocalLoss(w)
	loss := lossFn(x, 2, 2.0)

	assertEqualApprox(t, 0.36641273, loss.Value().Scalar())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.11375972, 0.12572393, -0.39304319, 0.15355955}, x.Grad().Data(), 1.0e-6)
}

func TestZeroOneQuantization(t *testing.T) {
	t.Run("float32", testZeroOneQuantization[float32])
	t.Run("float64", testZeroOneQuantization[float64])
}

func testZeroOneQuantization[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()
	x := g.NewVariable(mat.NewVecDense([]T{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), true)
	loss := ZeroOneQuantization(x)

	assertEqualApprox(t, 2.209, loss.Value().Scalar())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.144, 0.192, 0.0, 0.096, -7.488, 0.168}, x.Grad().Data(), 1.0e-6)
}

func TestNorm2Quantization(t *testing.T) {
	t.Run("float32", testNorm2Quantization[float32])
	t.Run("float64", testNorm2Quantization[float64])
}

func testNorm2Quantization[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()
	x := g.NewVariable(mat.NewVecDense([]T{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), true)
	loss := Norm2Quantization(x)

	assertEqualApprox(t, 0.8836, loss.Value().Scalar())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.376, 0.752, 3.76, 1.504, -3.008, 1.128}, x.Grad().Data(), 1.0e-6)
}

func TestOneHotQuantization(t *testing.T) {
	t.Run("float32", testOneHotQuantization[float32])
	t.Run("float64", testOneHotQuantization[float64])
}

func testOneHotQuantization[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()
	x := g.NewVariable(mat.NewVecDense([]T{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), true)
	loss := OneHotQuantization(x, 0.1)

	assertEqualApprox(t, 0.30926, loss.Value().Scalar())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{0.052, 0.0944, 0.376, 0.16, -1.0496, 0.1296}, x.Grad().Data(), 1.0e-6)
}

func TestMSESeqLoss(t *testing.T) {
	t.Run("float32", testMSESeqLoss[float32])
	t.Run("float64", testMSESeqLoss[float64])
}

func testMSESeqLoss[T mat.DType](t *testing.T) {
	g := ag.NewGraph[T]()
	x1 := g.NewVariable(mat.NewVecDense([]T{0.0, 0.1, 0.2, 0.3}), true)
	y1 := g.NewVariable(mat.NewVecDense([]T{0.3, 0.2, 0.1, 0.0}), false)
	x2 := g.NewVariable(mat.NewVecDense([]T{0.0, 0.1, 0.2, 0.3}), true)
	y2 := g.NewVariable(mat.NewVecDense([]T{0.3, 0.2, 0.1, 0.0}), false)
	loss := MSESeq([]ag.Node[T]{x1, x2}, []ag.Node[T]{y1, y2}, true)

	assertEqualApprox(t, 0.1, loss.Value().Scalar())

	ag.Backward(loss)

	assert.InDeltaSlice(t, []T{-0.15, -0.05, 0.05, 0.15}, x1.Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []T{-0.15, -0.05, 0.05, 0.15}, x2.Grad().Data(), 1.0e-6)
}

func assertEqualApprox[T mat.DType](t *testing.T, expected, actual T) {
	t.Helper()
	assert.InDelta(t, expected, actual, 1.0e-06)
}
