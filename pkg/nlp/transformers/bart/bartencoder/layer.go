// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartencoder

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
)

var (
	_ nn.Model     = &Layer{}
	_ nn.Processor = &LayerProcessor{}
)

type Layer struct {
	Config                 bartconfig.Config
	SelfAttention          *multiheadattention.Model
	SelfAttentionLayerNorm *layernorm.Model
	FFN                    *stack.Model // FC2(Activation(FC1))
	LayerNorm              *layernorm.Model
}

func NewLayer(config bartconfig.Config) *Layer {
	return &Layer{
		Config:                 config,
		SelfAttention:          multiheadattention.New(config.DModel, config.EncoderAttentionHeads, false), // TODO: config.AttentionDropout
		SelfAttentionLayerNorm: layernorm.New(config.DModel),
		FFN: stack.New(
			linear.New(config.DModel, config.EncoderFFNDim),
			activation.New(ag.OpGeLU), // TODO: config.ActivationFunction
			// dropout.New(config.ActivationDropout)
			linear.New(config.EncoderFFNDim, config.DModel),
			// dropout.New(config.Dropout)
		),
		LayerNorm: layernorm.New(config.DModel),
	}
}

type LayerProcessor struct {
	nn.BaseProcessor
	bartconfig.Config
	SelfAttention          *multiheadattention.Processor
	SelfAttentionLayerNorm *layernorm.Processor
	FFN                    *stack.Processor
	LayerNorm              *layernorm.Processor
}

func (m *Layer) NewProc(ctx nn.Context) nn.Processor {
	return &LayerProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		SelfAttention:          m.SelfAttention.NewProc(ctx).(*multiheadattention.Processor),
		SelfAttentionLayerNorm: m.SelfAttentionLayerNorm.NewProc(ctx).(*layernorm.Processor),
		FFN:                    m.FFN.NewProc(ctx).(*stack.Processor),
		LayerNorm:              m.LayerNorm.NewProc(ctx).(*layernorm.Processor),
	}
}

func (p *LayerProcessor) Forward(xs ...ag.Node) []ag.Node {
	selfAtt := p.selfAttentionBlock(xs)
	out := p.fullyConnectedBlock(selfAtt)
	// TODO: limit output values if any Inf or NaN
	return out
}

func (p *LayerProcessor) selfAttentionBlock(xs []ag.Node) []ag.Node {
	residual := p.copy(xs)
	if p.NormalizeBefore {
		xs = p.SelfAttentionLayerNorm.Forward(xs...)
	}
	xs = p.SelfAttention.Forward(xs...) //  query=x, key=x, key_padding_mask=encoder_padding_mask
	// xs = p.Dropout(xs) // config.Dropout
	xs = p.add(residual, xs)
	if !p.NormalizeBefore {
		xs = p.SelfAttentionLayerNorm.Forward(xs...)
	}
	return xs
}

func (p *LayerProcessor) fullyConnectedBlock(xs []ag.Node) []ag.Node {
	residual := p.copy(xs)
	if p.NormalizeBefore {
		xs = p.LayerNorm.Forward(xs...)
	}
	xs = p.FFN.Forward(xs...)
	xs = p.add(residual, xs)
	if !p.NormalizeBefore {
		xs = p.LayerNorm.Forward(xs...)
	}
	return xs
}

func (p *LayerProcessor) copy(xs []ag.Node) []ag.Node {
	copied := func(x ag.Node) ag.Node {
		return p.Graph.Identity(x)
	}
	return ag.Map(copied, xs)
}

func (p *LayerProcessor) add(a []ag.Node, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = p.Graph.Add(a[i], b[i])
	}
	return c
}
