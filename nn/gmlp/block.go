// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"

	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/sgu"
)

var _ nn.Model = &Block{}

// Block is the core model of the gMLP.
type Block struct {
	nn.Module
	Layers nn.ModuleList[nn.StandardModel]
}

// BlockConfig provides configuration parameters for a single Block of the gMLP Model.
type BlockConfig struct {
	Dim        int
	DimFF      int
	SeqLen     int
	Activation activation.Name
}

func init() {
	gob.Register(&Block{})
}

// NewBlock returns a new Block.
func NewBlock[T float.DType](config BlockConfig) *Block {
	return &Block{
		Layers: []nn.StandardModel{
			linear.New[T](config.Dim, config.DimFF),
			activation.New(activation.GELU),
			sgu.New[T](sgu.Config{
				Dim:        config.DimFF,
				DimSeq:     config.SeqLen,
				InitEps:    1e-3,
				Activation: config.Activation,
			}),
			linear.New[T](config.DimFF/2, config.Dim),
		},
	}
}

func (m *Block) Forward(xs ...ag.Node) []ag.Node {
	return m.Layers.Forward(xs...)
}
