// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fixnorm implements the fixnorm normalization method.
//
// Reference: "Improving Lexical Choice in Neural Machine Translation" by Toan Q. Nguyen and David Chiang (2018)
// (https://arxiv.org/pdf/1710.01329.pdf)
package fixnorm

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model is an empty model used to instantiate a new Processor.
type Model struct {
	nn.Module
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model.
func New() *Model {
	return &Model{}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...mat.Tensor) []mat.Tensor {
	if len(xs) == 0 {
		return nil
	}
	eps := xs[0].Value().(mat.Matrix).NewScalar(1e-10)
	ys := make([]mat.Tensor, len(xs))
	for i, x := range xs {
		norm := ag.Sqrt(ag.ReduceSum(ag.Square(x)))
		ys[i] = ag.DivScalar(x, ag.AddScalar(norm, eps))
	}
	return ys
}
