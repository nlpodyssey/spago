// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gmlp implements a model composed by basic MLP layers with gating mechanism.
// Reference: `Pay Attention to MLPs` by Liu et al, 2021 (https://arxiv.org/pdf/2105.08050.pdf)
package gmlp

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	Config Config
	Layers nn.ModuleList[nn.StandardModel]
}

// Config provides configuration parameters for a the gMLP Model.
type Config struct {
	Dim        int
	Depth      int
	SeqLen     int
	FFMult     int
	Activation activation.Name
	// TODO: ProbSurvival T
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Model.
func New[T float.DType](config Config) *Model {
	layers := make([]nn.StandardModel, config.Depth)
	for i := 0; i < config.Depth; i++ {
		layers[i] = NewResidual(
			NewPreNorm[T](
				config.Dim,
				NewBlock[T](BlockConfig{
					Dim:        config.Dim,
					DimFF:      config.Dim * config.FFMult,
					SeqLen:     config.SeqLen,
					Activation: config.Activation,
				}),
			),
		)
	}
	return &Model{
		Config: config,
		Layers: layers,
	}
}

// Forward performs the forward step. It adds pads if necessary.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	if len(xs) > m.Config.SeqLen {
		panic("gMLP: input sequence is too long")
	}
	if len(xs) == 0 {
		return nil
	}
	padded := ag.Pad(xs, m.Config.SeqLen, func(int) ag.Node {
		return xs[0].Value().NewEmptyVec(m.Config.Dim)
	})
	return m.Layers.Forward(padded...)
}
