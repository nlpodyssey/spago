// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gmlp implements a model composed by basic MLP layers with gating mechanism.
// Reference: `Pay Attention to MLPs` by Liu et al, 2021 (https://arxiv.org/pdf/2105.08050.pdf)
package gmlp

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/stack"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	Config Config
	*stack.Model[T]
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
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new Model.
func New[T mat.DType](config Config) *Model[T] {
	layer := func(_ int) nn.StandardModel[T] {
		return NewResidual(
			NewPreNorm(
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
	return &Model[T]{
		Config: config,
		Model:  stack.Make(config.Depth, layer), // TODO: add "prob to survive" in the `stack` pkg
	}
}

// Forward performs the forward step. It adds pads if necessary.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	if len(xs) > m.Config.SeqLen {
		panic("gMLP: input sequence is too long")
	}
	padded := ag.Pad(xs, m.Config.SeqLen, func(_ int) ag.Node[T] {
		return m.Session.Graph().NewVariable(mat.NewEmptyVecDense[T](m.Config.Dim), false)
	})
	return m.Model.Forward(padded...)
}
