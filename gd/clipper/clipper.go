// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package clipper

import (
	"math"

	"github.com/nlpodyssey/spago/mat"
)

// GradClipper is implemented by any value that has the Clip method.
type GradClipper[T mat.DType] interface {
	// Clip clips the values of the matrix in place.
	Clip(gs []mat.Matrix[T])
}

// ClipValue is a GradClipper which clips the values of a matrix between
// -Value and +Value.
type ClipValue[T mat.DType] struct {
	Value float64
}

// Clip clips the values of the matrix in place.
func (c *ClipValue[T]) Clip(gs []mat.Matrix[T]) {
	for _, g := range gs {
		g.ClipInPlace(-c.Value, c.Value)
	}
}

// ClipNorm is a GradClipper which clips the values of a matrix according to
// the NormType. See ClipNorm.Clip.
type ClipNorm[T mat.DType] struct {
	MaxNorm  float64
	NormType float64
}

// Clip clips the gradients, multiplying each parameter by the MaxNorm, divided by n-norm of the overall gradients.
// NormType is the n-norm. Can be ``Double.POSITIVE_INFINITY`` for infinity norm (default 2.0)
func (c *ClipNorm[T]) Clip(gs []mat.Matrix[T]) {
	if c.NormType <= 1 {
		panic("gd: norm type required to be > 1.")
	}

	var totalNorm float64
	if math.IsInf(c.NormType, 1) {
		for _, g := range gs {
			totalNorm = math.Max(g.Abs().Max().Scalar().Float64(), totalNorm)
		}
	} else {
		var sum float64
		for _, g := range gs {
			sum += g.Abs().Pow(c.NormType).Sum().Scalar().Float64()
		}
		totalNorm = math.Pow(sum, 1/c.NormType)
	}

	clipCoeff := c.MaxNorm / (totalNorm + 0.0000001)
	if clipCoeff < 1.0 {
		for _, g := range gs {
			g.ProdScalarInPlace(clipCoeff)
		}
	}
}
