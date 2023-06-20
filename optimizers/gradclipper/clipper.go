// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gradclipper

import (
	"context"
	"math"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

// GradClipper performs gradient clipping on a set of parameters.
type GradClipper interface {
	// ClipGrads clips the gradients in place.
	ClipGrads(parameters nn.ParamChannelFunc)
}

// ValueClipper is a GradClipper which clips the values of a matrix between -Value and +Value.
type ValueClipper struct {
	Value float64
}

// ClipGrads clips the gradients in place between -Value and +Value.
func (c *ValueClipper) ClipGrads(parameters nn.ParamChannelFunc) {
	for _, g := range collectGradients(parameters) {
		g.(mat.Matrix).ClipInPlace(-c.Value, c.Value)
	}
}

// NormClipper is a GradClipper which clips the values of a matrix according to the NormType.
type NormClipper struct {
	MaxNorm  float64
	NormType float64
}

// validateNormType ensures that the NormType is greater than 1.
func (c *NormClipper) validateNormType() {
	if c.NormType <= 1 {
		panic("gd: norm type required to be > 1.")
	}
}

// calculateTotalNorm calculates the total norm based on NormType and matrices gs.
func (c *NormClipper) calculateTotalNorm(gs []mat.Tensor) float64 {
	var totalNorm float64
	if math.IsInf(c.NormType, 1) {
		for _, g := range gs {
			totalNorm = math.Max(g.(mat.Matrix).Abs().Max().Item().F64(), totalNorm)
		}
	} else {
		var sum float64
		for _, g := range gs {
			sum += g.(mat.Matrix).Abs().Pow(c.NormType).Sum().Item().F64()
		}
		totalNorm = math.Pow(sum, 1/c.NormType)
	}
	return totalNorm
}

// ClipGradients clips the gradients, multiplying each parameter by the MaxNorm, divided by n-norm of the overall gradients.
// NormType is the n-norm. Can be “Double.POSITIVE_INFINITY“ for infinity norm (default 2.0)
func (c *NormClipper) ClipGradients(parameters nn.ParamChannelFunc) {
	grads := collectGradients(parameters)
	c.validateNormType()
	totalNorm := c.calculateTotalNorm(grads)

	clipCoeff := c.MaxNorm / (totalNorm + 1e-7)
	if clipCoeff < 1.0 {
		for _, g := range grads {
			g.(mat.Matrix).ProdScalarInPlace(clipCoeff)
		}
	}
}

// collectGradients collects all the gradients from the parameters channel and returns them as a slice.
func collectGradients(parameters nn.ParamChannelFunc) []mat.Tensor {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var allGrads []mat.Tensor
	for param := range parameters(ctx) {
		allGrads = append(allGrads, param.Grad())
	}
	return allGrads
}
