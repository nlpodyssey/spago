// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartdecoder

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartencoder"
)

var (
	_ nn.Model = &bartencoder.Layer{}
)

// Layer implements a BART decoder layer.
type Layer struct {
	nn.BaseModel
	Config                    bartconfig.Config
	SelfAttention             *multiheadattention.Model
	SelfAttentionLayerNorm    *layernorm.Model
	EncoderAttention          *multiheadattention.Model
	EncoderAttentionLayerNorm *layernorm.Model
	FFN                       *stack.Model
	LayerNorm                 *layernorm.Model
}

func init() {
	gob.Register(&Layer{})
}

// NewLayer returns a new BART decoder Layer.
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
			activation.New(ag.OpGELU), // TODO: config.ActivationFunction
			// dropout.New(config.ActivationDropout)
			linear.New(config.DecoderFFNDim, config.DModel),
			// dropout.New(config.Dropout)
		),
		LayerNorm: layernorm.New(config.DModel),
	}
}

// Forward performs the forward step for each input and returns the result.
// Valid input type: LayerInput.
func (m *Layer) Forward(xs, encoderHiddenStates []ag.Node) []ag.Node {
	selfAtt := m.selfAttentionBlock(xs)
	crossAtt := m.crossAttentionBlock(selfAtt, encoderHiddenStates)
	out := m.fullyConnectedBlock(crossAtt)
	return out
}

func (m *Layer) selfAttentionBlock(xs []ag.Node) []ag.Node {
	residual := m.copy(xs)
	if m.Config.NormalizeBefore {
		xs = m.SelfAttentionLayerNorm.Forward(xs...)
	}
	xs = m.SelfAttention.Forward(attention.ToQKV(xs))
	// TODO: xs = m.Dropout(xs)
	xs = m.add(residual, xs)
	if !m.Config.NormalizeBefore {
		xs = m.SelfAttentionLayerNorm.Forward(xs...)
	}
	return xs
}

func (m *Layer) crossAttentionBlock(xs []ag.Node, encoderHiddenStates []ag.Node) []ag.Node {
	residual := m.copy(xs)
	if m.Config.NormalizeBefore {
		xs = m.EncoderAttentionLayerNorm.Forward(xs...)
	}
	xs = m.EncoderAttention.Forward(attention.QKV{
		Queries: xs,
		Keys:    encoderHiddenStates,
		Values:  encoderHiddenStates,
	})
	// TODO: xs = m.Dropout(xs)
	xs = m.add(residual, xs)
	if !m.Config.NormalizeBefore {
		xs = m.EncoderAttentionLayerNorm.Forward(xs...)
	}
	return xs
}

func (m *Layer) fullyConnectedBlock(xs []ag.Node) []ag.Node {
	residual := m.copy(xs)
	if m.Config.NormalizeBefore {
		xs = m.LayerNorm.Forward(xs...)
	}
	xs = m.FFN.Forward(xs...)
	xs = m.add(residual, xs)
	if !m.Config.NormalizeBefore {
		xs = m.LayerNorm.Forward(xs...)
	}
	return xs
}

func (m *Layer) copy(xs []ag.Node) []ag.Node {
	copied := func(x ag.Node) ag.Node {
		return m.Graph().Identity(x)
	}
	return ag.Map(copied, xs)
}

func (m *Layer) add(a []ag.Node, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = m.Graph().Add(a[i], b[i])
	}
	return c
}
