// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package clipper

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"math"
)

type GradClipper interface {
	Clip(gs []mat.Matrix)
}

type ClipValue struct {
	Value float64
}

func (c *ClipValue) Clip(gs []mat.Matrix) {
	for _, g := range gs {
		g.ClipInPlace(-c.Value, c.Value)
	}
}

type ClipNorm struct {
	MaxNorm, NormType float64
}

// Clip clips the gradients, multiplying each parameter by the MaxNorm, divided by n-norm of the overall gradients.
// NormType is the n-norm. Can be ``Double.POSITIVE_INFINITY`` for infinity norm (default 2.0)
func (c *ClipNorm) Clip(gs []mat.Matrix) {
	if c.NormType <= 1 {
		panic("gd: norm type required to be > 1.")
	}

	totalNorm := 0.0
	if math.IsInf(c.NormType, 1) {
		for _, g := range gs {
			totalNorm = math.Max(g.Abs().Max(), totalNorm)
		}
	} else {
		sum := 0.0
		for _, g := range gs {
			sum += g.Abs().Pow(c.NormType).Sum()
		}
		totalNorm = math.Pow(sum, 1.0/c.NormType)
	}

	clipCoeff := c.MaxNorm / (totalNorm + 0.0000001)
	if clipCoeff < 1.0 {
		for _, g := range gs {
			g.ProdScalarInPlace(clipCoeff)
		}
	}
}
