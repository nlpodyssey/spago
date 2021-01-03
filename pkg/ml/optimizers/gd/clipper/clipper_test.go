// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package clipper

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestClipValue(t *testing.T) {
	gs := buildTestGrads()
	(&ClipValue{Value: 0.7}).Clip(gs)
	assert.InDeltaSlice(t, []mat.Float{
		0.5, 0.6, -0.7, -0.6,
		0.7, -0.4, 0.1, -0.7,
		0.7, -0.7, 0.3, 0.5,
		0.7, -0.7, 0.0, -0.1,
		0.4, 0.7, -0.7, 0.7,
	}, gs[0].Data(), 1.0e-05)
	assert.InDeltaSlice(t, []mat.Float{0.7, 0.7, 0.4, 0.7, 0.1}, gs[1].Data(), 1.0e-05)
}

func TestClip2Norm(t *testing.T) {
	gs := buildTestGrads()
	(&ClipNorm{MaxNorm: 2.0, NormType: 2.0}).Clip(gs)
	assert.InDeltaSlice(t, []mat.Float{
		0.314814, 0.377777, -0.503702, -0.377777,
		0.440739, -0.251851, 0.062962, -0.503702,
		0.440739, -0.440739, 0.188888, 0.314814,
		0.503702, -0.566665, 0.0, -0.062962,
		0.251851, 0.629628, -0.440739, 0.503702,
	}, gs[0].Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{0.566665, 0.440739, 0.251851, 0.503702, 0.062962}, gs[1].Data(), 1.0e-06)
}

func TestClipNormInf(t *testing.T) {
	gs := buildTestGrads()
	(&ClipNorm{MaxNorm: 0.5, NormType: mat.Inf(+1)}).Clip(gs)
	assert.InDeltaSlice(t, []mat.Float{
		0.25, 0.3, -0.4, -0.3,
		0.35, -0.2, 0.05, -0.4,
		0.35, -0.35, 0.15, 0.25,
		0.4, -0.45, 0.0, -0.05,
		0.2, 0.5, -0.35, 0.4,
	}, gs[0].Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{0.45, 0.35, 0.2, 0.4, 0.05}, gs[1].Data(), 1.0e-06)
}

func buildTestGrads() []mat.Matrix {
	return []mat.Matrix{
		mat.NewDense(4, 5, []mat.Float{
			0.5, 0.6, -0.8, -0.6,
			0.7, -0.4, 0.1, -0.8,
			0.7, -0.7, 0.3, 0.5,
			0.8, -0.9, 0.0, -0.1,
			0.4, 1.0, -0.7, 0.8,
		}),
		mat.NewVecDense([]mat.Float{0.9, 0.7, 0.4, 0.8, 0.1}),
	}
}
