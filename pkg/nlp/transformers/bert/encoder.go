// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Reference: "Attention Is All You Need" by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin (2017)
(http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf).
*/
package bert

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model     = &Encoder{}
	_ nn.Processor = &EncoderProcessor{}
)

// TODO: include and use the dropout hyper-parameter
type EncoderConfig struct {
	Size                   int
	NumOfAttentionHeads    int
	IntermediateSize       int
	IntermediateActivation ag.OpName
	NumOfLayers            int
}

type Encoder struct {
	EncoderConfig
	*stack.Model
}

// LayerAt returns the i-layer model.
func (m *Encoder) LayerAt(i int) *EncoderLayer {
	return m.Layers[i].(*EncoderLayer)
}

type EncoderProcessor struct {
	*stack.Processor
}

func (m *Encoder) NewProc(ctx nn.Context) nn.Processor {
	return &EncoderProcessor{
		Processor: m.Model.NewProc(ctx).(*stack.Processor),
	}
}

// LayerProcAt returns the i-processor.
// It panics if the underlying model is not BERT.
func (p *EncoderProcessor) LayerAt(index int) *EncoderLayerProcessor {
	return p.Layers[index].(*EncoderLayerProcessor)
}

// NewBertEncoder returns a new BERT encoder model composed of a stack of N identical BERT encoder layers.
func NewBertEncoder(config EncoderConfig) *Encoder {
	layers := make([]nn.Model, config.NumOfLayers)
	for layerIndex := range layers {
		layers[layerIndex] = &EncoderLayer{
			MultiHeadAttention: multiheadattention.New(config.Size, config.NumOfAttentionHeads),
			NormAttention:      layernorm.New(config.Size),
			FFN: stack.New(
				linear.New(config.Size, config.IntermediateSize),
				activation.New(config.IntermediateActivation),
				linear.New(config.IntermediateSize, config.Size),
			),
			NormFFN: layernorm.New(config.Size),
			Index:   layerIndex,
		}
	}
	return &Encoder{
		EncoderConfig: config,
		Model:         stack.New(layers...),
	}
}

// NewAlbertEncoder returns a new variant of the BERT encoder model.
// In this variant the stack of N identical BERT encoder layers share the same parameters.
func NewAlbertEncoder(config EncoderConfig) *Encoder {
	sharedLayer := &EncoderLayer{
		MultiHeadAttention: multiheadattention.New(config.Size, config.NumOfAttentionHeads),
		NormAttention:      layernorm.New(config.Size),
		FFN: stack.New(
			linear.New(config.Size, config.IntermediateSize),
			activation.New(config.IntermediateActivation),
			linear.New(config.IntermediateSize, config.Size),
		),
		NormFFN: layernorm.New(config.Size),
	}
	layers := make([]nn.Model, config.NumOfLayers)
	for layerIndex := range layers {
		layers[layerIndex] = sharedLayer
	}
	return &Encoder{
		EncoderConfig: config,
		Model:         stack.New(layers...),
	}
}
