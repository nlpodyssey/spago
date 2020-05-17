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
	"github.com/nlpodyssey/spago/pkg/ml/nn/rc"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Model     = &Layer{}
	_ nn.Processor = &LayerProcessor{}
)

// TODO: include and use the dropout hyper-parameter
type Config struct {
	Size                   int
	NumOfAttentionHeads    int
	IntermediateSize       int
	IntermediateActivation ag.OpName
	NumOfLayers            int
}

type Model struct {
	Config
	*stack.Model
}

// New returns a new BERT model composed of a stack of N identical BERT layers.
func New(config Config) *Model {
	layers := make([]nn.Model, config.NumOfLayers)
	for layerIndex := range layers {
		layers[layerIndex] = &Layer{
			MultiHeadAttention: multiheadattention.New(config.Size, config.NumOfAttentionHeads),
			NormAttention:      layernorm.New(config.Size),
			FFN: stack.New(
				linear.New(config.Size, config.IntermediateSize),
				activation.New(config.IntermediateActivation),
				linear.New(config.IntermediateSize, config.Size),
			),
			NormFFN: layernorm.New(config.Size),
		}
	}
	return &Model{
		Config: config,
		Model:  stack.New(layers...),
	}
}

// NewALBERT returns a new BERT model composed of a stack of N identical BERT layers, sharing the same parameters.
func NewALBERT(config Config) *Model {
	sharedLayer := &Layer{
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
	return &Model{
		Config: config,
		Model:  stack.New(layers...),
	}
}

// LayerAt returns the i-layer model.
func (m *Model) LayerAt(i int) *Layer {
	return m.Layers[i].(*Layer)
}

// LayerProcAt returns the i-processor.
// It panics if the underlying model is not BERT.
func LayerProcAt(bertProc *stack.Processor, index int) *LayerProcessor {
	if _, ok := bertProc.GetModel().(*Model); ok {
		return bertProc.Layers[index].(*LayerProcessor)
	} else {
		panic("bert: invalid neural model")
	}
}

// Single BERT Layer.
type Layer struct {
	MultiHeadAttention *multiheadattention.Model
	NormAttention      *layernorm.Model
	FFN                *stack.Model
	NormFFN            *layernorm.Model
}

type LayerProcessor struct {
	nn.BaseProcessor
	MultiHeadAttention *multiheadattention.Processor
	NormAttention      nn.Processor
	FFN                nn.Processor
	NormFFN            nn.Processor
}

func (m *Layer) NewProc(g *ag.Graph) nn.Processor {
	return &LayerProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		MultiHeadAttention: m.MultiHeadAttention.NewProc(g).(*multiheadattention.Processor),
		NormAttention:      m.NormAttention.NewProc(g),
		FFN:                m.FFN.NewProc(g),
		NormFFN:            m.NormFFN.NewProc(g),
	}
}

func (p *LayerProcessor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	nn.SetProcessingMode(mode, p.MultiHeadAttention, p.NormAttention, p.FFN, p.NormFFN)
}

func (p *LayerProcessor) Forward(xs ...ag.Node) []ag.Node {
	subLayer1 := rc.PostNorm(p.Graph, p.MultiHeadAttention.Forward, p.NormAttention.Forward, xs...)
	subLayer2 := rc.PostNorm(p.Graph, p.FFN.Forward, p.NormFFN.Forward, subLayer1...)
	return subLayer2
}
