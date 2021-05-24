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
	"github.com/nlpodyssey/spago/pkg/ml/nn/sgu"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var _ nn.Model = &Block{}

// Block is the core model of the gMLP.
type Block struct {
	nn.BaseModel
	*stack.Model
}

// BlockConfig provides configuration parameters for a single Block of the gMLP Model.
type BlockConfig struct {
	Dim        int
	DimFF      int
	SeqLen     int
	Activation ag.OpName
}

func init() {
	gob.Register(&Block{})
}

// NewBlock returns a new Block.
func NewBlock(config BlockConfig) *Block {
	return &Block{
		Model: stack.New(
			linear.New(config.Dim, config.DimFF),
			activation.New(ag.OpGELU),
			sgu.New(sgu.Config{
				Dim:        config.DimFF,
				DimSeq:     config.SeqLen,
				InitEps:    1e-3,
				Activation: config.Activation,
			}),
			linear.New(config.DimFF/2, config.Dim),
		),
	}
}
