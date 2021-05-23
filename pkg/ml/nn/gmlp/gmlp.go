// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	SeqLen       int
	ProbSurvival mat.Float
	ToEmbed      *embeddings.Model
	Layers       []*Residual
	ToLogits     *stack.Model
}

var _ nn.Model = &Model{}

type Config struct {
	// Set NumTokens <= 0 to disable embeddings.
	NumTokens int
	Dim       int
	Depth     int
	SeqLen    int
	FFMult    int
	// Set AttnDim <= 0 to disable attention.
	AttnDim      int
	ProbSurvival mat.Float
	Causal       bool
	// EmbeddingsDBPath is ignored if NumTokens <= 0.
	EmbeddingsDBPath string
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Model.
func New(config Config) *Model {
	return &Model{
		SeqLen:       config.SeqLen,
		ProbSurvival: config.ProbSurvival,
		ToEmbed:      newToEmbed(config),
		Layers:       newLayers(config),
		ToLogits:     newToLogits(config),
	}
}

func newLayers(config Config) []*Residual {
	layers := make([]*Residual, config.Depth)
	for i := range layers {
		layers[i] = newLayer(config)
	}
	return layers
}

func newLayer(config Config) *Residual {
	return NewResidual(
		NewPreNorm(
			config.Dim,
			NewGMLPBlock(GMLPBlockConfig{
				Dim:     config.Dim,
				DimFF:   config.Dim * config.FFMult,
				SeqLen:  config.SeqLen,
				AttnDim: config.AttnDim,
				Causal:  config.Causal,
			}),
		),
	)
}

func newToLogits(config Config) *stack.Model {
	if config.NumTokens <= 0 {
		return nil // TODO: else nn.Identity() ?
	}
	return stack.New(
		layernorm.New(config.Dim),
		linear.New(config.Dim, config.NumTokens),
	)
}

func newToEmbed(config Config) *embeddings.Model {
	if config.NumTokens <= 0 {
		return nil // TODO: else nn.Identity() ?
	}
	return embeddings.New(embeddings.Config{
		Size:             config.Dim,
		UseZeroEmbedding: true, // TODO: verify...
		DBPath:           config.EmbeddingsDBPath,
		ReadOnly:         false, // TODO: verify... depending on training?
		ForceNewDB:       true,  // TODO: verify...
	})
}

func (m *Model) ForwardWords(words []string) []ag.Node {
	if m.ToEmbed == nil || m.ToLogits == nil {
		panic("gMLP: cannot forward words without ToEmbed or ToLogits")
	}
	xs := m.ToEmbed.Encode(words)
	out := m.Forward(xs...)
	return m.ToLogits.Forward(out...)
}

func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	//if m.Mode() == nn.Training {
	//	return m.forwardTraining(xs...)
	//}

	ys := xs
	for _, layer := range m.Layers {
		ys = layer.Forward(ys...)
	}
	return ys
}

// TODO: implement Model.forwardTraining
func (m *Model) forwardTraining(xs ...ag.Node) []ag.Node {
	panic("Model.forwardTraining is not implemented")
}
