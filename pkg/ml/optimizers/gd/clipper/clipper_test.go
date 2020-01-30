// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package clipper

import (
	"gonum.org/v1/gonum/floats"
	"math"
	"saientist.dev/spago/pkg/mat"
	"testing"
)

func TestClipValue(t *testing.T) {
	gs := buildTestGrads()
	(&ClipValue{value: 0.7}).Clip(gs)
	if !floats.EqualApprox(gs[0].Data(), []float64{
		0.5, 0.6, -0.7, -0.6,
		0.7, -0.4, 0.1, -0.7,
		0.7, -0.7, 0.3, 0.5,
		0.7, -0.7, 0.0, -0.1,
		0.4, 0.7, -0.7, 0.7,
	}, 1.0e-05) {
		t.Error("gd: gs[0] doesn't match the expected values")
	}
	if !floats.EqualApprox(gs[1].Data(), []float64{0.7, 0.7, 0.4, 0.7, 0.1}, 1.0e-05) {
		t.Error("gd: gs[1] doesn't match the expected values")
	}
}

func TestClip2Norm(t *testing.T) {
	gs := buildTestGrads()
	(&ClipNorm{MaxNorm: 2.0, NormType: 2.0}).Clip(gs)
	if !floats.EqualApprox(gs[0].Data(), []float64{
		0.314814, 0.377777, -0.503702, -0.377777,
		0.440739, -0.251851, 0.062962, -0.503702,
		0.440739, -0.440739, 0.188888, 0.314814,
		0.503702, -0.566665, 0.0, -0.062962,
		0.251851, 0.629628, -0.440739, 0.503702,
	}, 1.0e-06) {
		t.Error("gd: gs[0] doesn't match the expected values")
	}
	if !floats.EqualApprox(gs[1].Data(), []float64{0.566665, 0.440739, 0.251851, 0.503702, 0.062962}, 1.0e-06) {
		t.Error("gd: gs[1] doesn't match the expected values")
	}
}

func TestClipNormInf(t *testing.T) {
	gs := buildTestGrads()
	(&ClipNorm{MaxNorm: 0.5, NormType: math.Inf(+1)}).Clip(gs)
	if !floats.EqualApprox(gs[0].Data(), []float64{
		0.25, 0.3, -0.4, -0.3,
		0.35, -0.2, 0.05, -0.4,
		0.35, -0.35, 0.15, 0.25,
		0.4, -0.45, 0.0, -0.05,
		0.2, 0.5, -0.35, 0.4,
	}, 1.0e-06) {
		t.Error("gd: gs[0] doesn't match the expected values")
	}
	if !floats.EqualApprox(gs[1].Data(), []float64{0.45, 0.35, 0.2, 0.4, 0.05}, 1.0e-06) {
		t.Error("gd: gs[1] doesn't match the expected values")
	}
}

func buildTestGrads() []mat.Matrix {
	return []mat.Matrix{
		mat.NewDense(4, 5, []float64{
			0.5, 0.6, -0.8, -0.6,
			0.7, -0.4, 0.1, -0.8,
			0.7, -0.7, 0.3, 0.5,
			0.8, -0.9, 0.0, -0.1,
			0.4, 1.0, -0.7, 0.8,
		}),
		mat.NewVecDense([]float64{0.9, 0.7, 0.4, 0.8, 0.1}),
	}
}
