// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartdecoder

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartencoder"
)

var (
	_ nn.Model     = &bartencoder.Layer{}
	_ nn.Processor = &bartencoder.LayerProcessor{}
)

type Layer struct {
	Config                    bartconfig.Config
	SelfAttention             *multiheadattention.Model
	SelfAttentionLayerNorm    *layernorm.Model
	EncoderAttention          *multiheadattention.Model
	EncoderAttentionLayerNorm *layernorm.Model
	FFN                       *stack.Model // FC2(Activation(FC1))
	LayerNorm                 *layernorm.Model
}

func NewLayer(config bartconfig.Config) *Layer {
	return &Layer{
		Config: config,
		SelfAttention: multiheadattention.New(
			config.DModel,
			config.DecoderAttentionHeads,
			true, // use causal mask
			// TODO: config.AttentionDropout
		),
		SelfAttentionLayerNorm: layernorm.New(config.DModel),
		EncoderAttention: multiheadattention.New(
			config.DModel,
			config.DecoderAttentionHeads,
			false, // don't use causal mask
			// TODO: config.AttentionDropout, encoder_decoder_attention=True
		),
		EncoderAttentionLayerNorm: layernorm.New(config.DModel),
		FFN: stack.New(
			linear.New(config.DModel, config.DecoderFFNDim),
			activation.New(ag.OpGeLU), // TODO: config.ActivationFunction
			// dropout.New(config.ActivationDropout)
			linear.New(config.DecoderFFNDim, config.DModel),
			// dropout.New(config.Dropout)
		),
		LayerNorm: layernorm.New(config.DModel),
	}
}

type LayerProcessor struct {
	nn.BaseProcessor
	bartconfig.Config
	SelfAttention             *multiheadattention.Processor
	SelfAttentionLayerNorm    *layernorm.Processor
	EncoderAttention          *multiheadattention.Processor
	EncoderAttentionLayerNorm *layernorm.Processor
	FFN                       *stack.Processor
	LayerNorm                 *layernorm.Processor
}

func (m *Layer) NewProc(ctx nn.Context) nn.Processor {
	return &LayerProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		SelfAttention:             m.SelfAttention.NewProc(ctx).(*multiheadattention.Processor),
		SelfAttentionLayerNorm:    m.SelfAttentionLayerNorm.NewProc(ctx).(*layernorm.Processor),
		EncoderAttention:          m.EncoderAttention.NewProc(ctx).(*multiheadattention.Processor),
		EncoderAttentionLayerNorm: m.EncoderAttentionLayerNorm.NewProc(ctx).(*layernorm.Processor),
		FFN:                       m.FFN.NewProc(ctx).(*stack.Processor),
		LayerNorm:                 m.LayerNorm.NewProc(ctx).(*layernorm.Processor),
	}
}

func (p *LayerProcessor) Process(xs []ag.Node, encoderHiddenStates []ag.Node) []ag.Node {
	selfAtt := p.selfAttentionBlock(xs)
	crossAtt := p.crossAttentionBlock(selfAtt, encoderHiddenStates)
	out := p.fullyConnectedBlock(crossAtt)
	return out
}

func (p *LayerProcessor) selfAttentionBlock(xs []ag.Node) []ag.Node {
	residual := p.copy(xs)
	if p.NormalizeBefore {
		xs = p.SelfAttentionLayerNorm.Forward(xs...)
	}
	xs = p.SelfAttention.Forward(xs...) // query=xs, key=xs, value=xs
	// xs = p.Dropout(xs)
	xs = p.add(residual, xs)
	if !p.NormalizeBefore {
		xs = p.SelfAttentionLayerNorm.Forward(xs...)
	}
	return xs
}

func (p *LayerProcessor) crossAttentionBlock(xs []ag.Node, ex []ag.Node) []ag.Node {
	residual := p.copy(xs)
	if p.NormalizeBefore {
		xs = p.EncoderAttentionLayerNorm.Forward(xs...)
	}
	xs = p.EncoderAttention.ForwardQKV(xs, ex, ex) // query=xs, key=ex, value=ex
	// xs = p.Dropout(xs)
	xs = p.add(residual, xs)
	if !p.NormalizeBefore {
		xs = p.EncoderAttentionLayerNorm.Forward(xs...)
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

func (p *LayerProcessor) Forward(_ ...ag.Node) []ag.Node {
	panic("bertdecoder: Forward() not implemented; use Process() instead.")
}
