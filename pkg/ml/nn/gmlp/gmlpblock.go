// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

type GMLPBlock struct {
	nn.BaseModel
	*stack.Model
}

var _ nn.Model = &GMLPBlock{}

// GMLPBlockConfig provides configuration parameters for GMLPBlock.
type GMLPBlockConfig struct {
	Dim    int
	DimFF  int
	SeqLen int
	// Set AttnDim <= 0 to disable attention.
	AttnDim int
	Causal  bool
}

func init() {
	gob.Register(&GMLPBlock{})
}

// NewGMLPBlock returns a new GMLPBlock.
func NewGMLPBlock(config GMLPBlockConfig) *GMLPBlock {
	return &GMLPBlock{
		Model: stack.New(
			linear.New(config.Dim, config.DimFF),
			activation.New(ag.OpGELU),
			NewSpatialGatingUnit(SpatialGatingUnitConfig{
				Dim:     config.DimFF,
				DimSeq:  config.SeqLen,
				AttnDim: config.AttnDim,
				Causal:  config.Causal,
				InitEps: 1e-3,
			}),
			linear.New(config.DimFF/2, config.Dim),
		),
	}
}
