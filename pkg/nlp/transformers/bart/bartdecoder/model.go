// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartdecoder

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/posembeddings"
	"github.com/nlpodyssey/spago/pkg/utils"
)

var (
	_ nn.Model = &Model{}
)

// Model implements a BART decoder.
type Model struct {
	nn.BaseModel
	Config                      bartconfig.Config
	LearnedPositionalEmbeddings *posembeddings.LearnedPositionalEmbeddings
	Layers                      *stack.Model
	EmbeddingLayerNorm          *layernorm.Model
	LayerNorm                   *layernorm.Model
}

// ModelInput is a set of values suitable as input for the forward step of a BART Decoder Model.
type ModelInput struct {
	Xs      []ag.Node
	Encoded []ag.Node
}

// New returns a new BART decoder Model.
func New(config bartconfig.Config) *Model {
	if config.StaticPositionEmbeddings {
		panic("bart: static position embeddings not implemented.")
	}
	if config.ScaleEmbedding {
		panic("bart: scale embedding not implemented.")
	}
	learnedPositionalEmbeddings := make([]nn.Param, config.MaxPositionEmbeddings+config.ExtraPosEmbedding)
	for i := 0; i < len(learnedPositionalEmbeddings); i++ {
		learnedPositionalEmbeddings[i] = nn.NewParam(mat.NewEmptyVecDense(config.DModel))
	}
	return &Model{
		BaseModel: nn.BaseModel{RCS: true},
		Config:    config,
		LearnedPositionalEmbeddings: posembeddings.NewLearnedPositionalEmbeddings(
			posembeddings.Config{
				NumEmbeddings: config.VocabSize,
				EmbeddingDim:  config.DModel,
				PaddingIDX:    0, // TODO
				Offset:        config.ExtraPosEmbedding,
			}),
		EmbeddingLayerNorm: layernorm.New(config.DModel),
		Layers: stack.Make(config.DecoderLayers, func(_ int) nn.Model {
			return NewLayer(config)
			// add LayerDrop to skip layers during training (see https://arxiv.org/abs/1909.11556 for description)
		}),
		LayerNorm: layernorm.New(config.DModel),
	}
}

// Forward performs the forward step for each input and returns the result.
// Valid input type: ModelInput.
func (m *Model) Forward(in interface{}) interface{} {
	mi := in.(ModelInput)
	//func (m *Model) Decode(xs, encoded []ag.Node) []ag.Node {
	embedPos := m.LearnedPositionalEmbeddings.Forward(utils.MakeIndices(len(mi.Xs))).([]ag.Node)
	ys := m.add(mi.Xs, embedPos)
	ys = m.EmbeddingLayerNorm.Forward(ys).([]ag.Node)
	// ys = m.Dropout(ys)

	for _, layer := range m.Layers.Layers {
		ys = layer.(*Layer).Forward(LayerInput{
			Xs:                  ys,
			EncoderHiddenStates: mi.Encoded,
		}).([]ag.Node)
		// TODO: save all hidden states into the processor to allow a later access
	}

	if m.Config.FinalLayerNorm {
		ys = m.LayerNorm.Forward(ys).([]ag.Node)
	}
	return ys
}

func (m *Model) add(a []ag.Node, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = m.Graph().Add(a[i], b[i])
	}
	return c
}
