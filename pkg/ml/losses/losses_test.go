// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package losses

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMSELoss(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]mat.Float{0.0, 0.1, 0.2, 0.3}), true)
	y := g.NewVariable(mat.NewVecDense([]mat.Float{0.3, 0.2, 0.1, 0.0}), false)
	loss := MSE(g, x, y, false)

	assertEqualApprox(t, 0.1, loss.Value().Scalar())

	g.Backward(loss)

	assert.InDeltaSlice(t, []mat.Float{-0.3, -0.1, 0.1, 0.3}, x.Grad().Data(), 1.0e-6)
}

func TestNLLLoss(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]mat.Float{-500, 0, 0.693147, 1.94591}), true)
	y := g.NewVariable(mat.NewVecDense([]mat.Float{0.0, 0.0, 1.0, 0.0}), false)
	loss := NLL(g, g.Softmax(x), y)

	assertEqualApprox(t, 1.609438, loss.Value().Scalar())

	g.Backward(loss)

	assert.InDeltaSlice(t, []mat.Float{0.0, 0.1, -0.8, 0.7}, x.Grad().Data(), 1.0e-6)
}

func TestCrossEntropyLoss(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]mat.Float{-500, 0, 0.693147, 1.94591}), true)
	loss := CrossEntropy(g, x, 2)

	assertEqualApprox(t, 1.609438, loss.Value().Scalar())

	g.Backward(loss)

	assert.InDeltaSlice(t, []mat.Float{0.0, 0.1, -0.8, 0.7}, x.Grad().Data(), 1.0e-6)
}

func TestZeroOneQuantization(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]mat.Float{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), true)
	loss := ZeroOneQuantization(g, x)

	assertEqualApprox(t, 2.209, loss.Value().Scalar())

	g.Backward(loss)

	assert.InDeltaSlice(t, []mat.Float{0.144, 0.192, 0.0, 0.096, -7.488, 0.168}, x.Grad().Data(), 1.0e-6)
}

func TestNorm2Quantization(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]mat.Float{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), true)
	loss := Norm2Quantization(g, x)

	assertEqualApprox(t, 0.8836, loss.Value().Scalar())

	g.Backward(loss)

	assert.InDeltaSlice(t, []mat.Float{0.376, 0.752, 3.76, 1.504, -3.008, 1.128}, x.Grad().Data(), 1.0e-6)
}

func TestOneHotQuantization(t *testing.T) {
	g := ag.NewGraph()
	x := g.NewVariable(mat.NewVecDense([]mat.Float{0.1, 0.2, 1.0, 0.4, -0.8, 0.3}), true)
	loss := OneHotQuantization(g, x, 0.1)

	assertEqualApprox(t, 0.30926, loss.Value().Scalar())

	g.Backward(loss)

	assert.InDeltaSlice(t, []mat.Float{0.052, 0.0944, 0.376, 0.16, -1.0496, 0.1296}, x.Grad().Data(), 1.0e-6)
}

func TestMSESeqLoss(t *testing.T) {
	g := ag.NewGraph()
	x1 := g.NewVariable(mat.NewVecDense([]mat.Float{0.0, 0.1, 0.2, 0.3}), true)
	y1 := g.NewVariable(mat.NewVecDense([]mat.Float{0.3, 0.2, 0.1, 0.0}), false)
	x2 := g.NewVariable(mat.NewVecDense([]mat.Float{0.0, 0.1, 0.2, 0.3}), true)
	y2 := g.NewVariable(mat.NewVecDense([]mat.Float{0.3, 0.2, 0.1, 0.0}), false)
	loss := MSESeq(g, []ag.Node{x1, x2}, []ag.Node{y1, y2}, true)

	assertEqualApprox(t, 0.1, loss.Value().Scalar())

	g.Backward(loss)

	assert.InDeltaSlice(t, []mat.Float{-0.15, -0.05, 0.05, 0.15}, x1.Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []mat.Float{-0.15, -0.05, 0.05, 0.15}, x2.Grad().Data(), 1.0e-6)
}

func assertEqualApprox(t *testing.T, expected, actual mat.Float) {
	t.Helper()
	assert.InDelta(t, expected, actual, 1.0e-06)
}
