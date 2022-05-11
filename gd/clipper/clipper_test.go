// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package clipper

import (
	"math"
	"testing"

	"github.com/nlpodyssey/spago/mat"
	"github.com/stretchr/testify/assert"
)

func TestClipValue(t *testing.T) {
	t.Run("float32", testClipValue[float32])
	t.Run("float64", testClipValue[float64])
}

func testClipValue[T mat.DType](t *testing.T) {
	gs := buildTestGrads[T]()
	(&ClipValue[T]{Value: 0.7}).Clip(gs)
	assert.InDeltaSlice(t, []T{
		0.5, 0.6, -0.7, -0.6,
		0.7, -0.4, 0.1, -0.7,
		0.7, -0.7, 0.3, 0.5,
		0.7, -0.7, 0.0, -0.1,
		0.4, 0.7, -0.7, 0.7,
	}, gs[0].Data(), 1.0e-05)
	assert.InDeltaSlice(t, []T{0.7, 0.7, 0.4, 0.7, 0.1}, gs[1].Data(), 1.0e-05)
}

func TestClip2Norm(t *testing.T) {
	t.Run("float32", testClip2Norm[float32])
	t.Run("float64", testClip2Norm[float64])
}

func testClip2Norm[T mat.DType](t *testing.T) {
	gs := buildTestGrads[T]()
	(&ClipNorm[T]{MaxNorm: 2.0, NormType: 2.0}).Clip(gs)
	assert.InDeltaSlice(t, []T{
		0.314814, 0.377777, -0.503702, -0.377777,
		0.440739, -0.251851, 0.062962, -0.503702,
		0.440739, -0.440739, 0.188888, 0.314814,
		0.503702, -0.566665, 0.0, -0.062962,
		0.251851, 0.629628, -0.440739, 0.503702,
	}, gs[0].Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{0.566665, 0.440739, 0.251851, 0.503702, 0.062962}, gs[1].Data(), 1.0e-06)
}

func TestClipNormInf(t *testing.T) {
	t.Run("float32", testClipNormInf[float32])
	t.Run("float64", testClipNormInf[float64])
}

func testClipNormInf[T mat.DType](t *testing.T) {
	gs := buildTestGrads[T]()
	(&ClipNorm[T]{MaxNorm: 0.5, NormType: math.Inf(1)}).Clip(gs)
	assert.InDeltaSlice(t, []T{
		0.25, 0.3, -0.4, -0.3,
		0.35, -0.2, 0.05, -0.4,
		0.35, -0.35, 0.15, 0.25,
		0.4, -0.45, 0.0, -0.05,
		0.2, 0.5, -0.35, 0.4,
	}, gs[0].Data(), 1.0e-06)
	assert.InDeltaSlice(t, []T{0.45, 0.35, 0.2, 0.4, 0.05}, gs[1].Data(), 1.0e-06)
}

func buildTestGrads[T mat.DType]() []mat.Matrix[T] {
	return []mat.Matrix[T]{
		mat.NewDense(4, 5, []T{
			0.5, 0.6, -0.8, -0.6,
			0.7, -0.4, 0.1, -0.8,
			0.7, -0.7, 0.3, 0.5,
			0.8, -0.9, 0.0, -0.1,
			0.4, 1.0, -0.7, 0.8,
		}),
		mat.NewVecDense([]T{0.9, 0.7, 0.4, 0.8, 0.1}),
	}
}
