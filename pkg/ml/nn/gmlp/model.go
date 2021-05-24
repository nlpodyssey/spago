// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gmlp implements a model composed by basic MLP layers with gating mechanism.
// Reference: `Pay Attention to MLPs` by Liu et al, 2021 (https://arxiv.org/pdf/2105.08050.pdf)
package gmlp

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	Config Config
	*stack.Model
}

// Config provides configuration parameters for a the gMLP Model.
type Config struct {
	Dim    int
	Depth  int
	SeqLen int
	FFMult int
	// TODO: ProbSurvival mat.Float
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Model.
func New(config Config) *Model {
	layer := func(_ int) nn.StandardModel {
		return NewResidual(
			NewPreNorm(
				config.Dim,
				NewBlock(BlockConfig{
					Dim:    config.Dim,
					DimFF:  config.Dim * config.FFMult,
					SeqLen: config.SeqLen,
				}),
			),
		)
	}
	return &Model{
		Config: config,
		Model:  stack.Make(config.Depth, layer), // TODO: add "prob to survive" in the `stack` pkg
	}
}

// Forward performs the forward step. It adds pads if necessary.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	// TODO: add padding
	return m.Model.Forward(xs...)
}
