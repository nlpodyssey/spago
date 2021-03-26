// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layer

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
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/encoder/layer"
)

var (
	_ nn.Model = &layer.Layer{}
)

// Layer implements a BART decoder layer.
type Layer struct {
	nn.BaseModel
	Config                    config.Config
	SelfAttention             *multiheadattention.Model
	SelfAttentionLayerNorm    *layernorm.Model
	EncoderAttention          *multiheadattention.Model
	EncoderAttentionLayerNorm *layernorm.Model
	FFN                       *stack.Model
	LayerNorm                 *layernorm.Model
}

func init() {
	gob.RegisterName("*bart.decoder.layer.Layer", &Layer{})
}

// NewLayer returns a new BART decoder Layer.
func NewLayer(config config.Config) *Layer {
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
			activation.New(mustGetOpName(config.ActivationFunction)),
			// dropout.New(config.ActivationDropout)
			linear.New(config.DecoderFFNDim, config.DModel),
			// dropout.New(config.Dropout)
		),
		LayerNorm: layernorm.New(config.DModel),
	}
}

func mustGetOpName(str string) ag.OpName {
	value, err := ag.GetOpName(str)
	if err != nil {
		panic(err)
	}
	return value
}

// KeysValuesPairs contains the keys and values used by the self-attention and cross-attention blocks.
type KeysValuesPairs struct {
	// SelfAttKeyValues contains the keys and values used by self-attention.
	SelfAttKeyValues multiheadattention.KeysValuesPairs
	// CrossAttKeyValues contains the keys and values used by cross-attention.
	CrossAttKeyValues multiheadattention.KeysValuesPairs
}

// Forward performs the forward step for each input and returns the result.
func (m *Layer) Forward(
	xs []ag.Node,
	encoderHiddenStates []ag.Node,
	pastProjKeysValues KeysValuesPairs,
) ([]ag.Node, KeysValuesPairs) {
	selfAtt, selfAttKeyValues := m.selfAttentionBlock(xs, pastProjKeysValues.SelfAttKeyValues)
	crossAtt, crossAttKeyValues := m.crossAttentionBlock(selfAtt, encoderHiddenStates, pastProjKeysValues.CrossAttKeyValues)
	out := m.fullyConnectedBlock(crossAtt)

	return out, KeysValuesPairs{
		SelfAttKeyValues:  selfAttKeyValues,
		CrossAttKeyValues: crossAttKeyValues,
	}
}

func (m *Layer) selfAttentionBlock(
	xs []ag.Node,
	pastProjKeysValues multiheadattention.KeysValuesPairs,
) ([]ag.Node, multiheadattention.KeysValuesPairs) {
	residual := m.copy(xs)
	if m.Config.NormalizeBefore {
		xs = m.SelfAttentionLayerNorm.Forward(xs...)
	}
	att := m.SelfAttention.ForwardWithPastKeysValues(attention.ToQKV(xs), pastProjKeysValues)
	xs = att.AttOutput
	// TODO: xs = m.Dropout(xs)
	xs = m.add(residual, xs)
	if !m.Config.NormalizeBefore {
		xs = m.SelfAttentionLayerNorm.Forward(xs...)
	}
	return xs, att.ProjKeysValues
}

func (m *Layer) crossAttentionBlock(
	xs []ag.Node,
	encoderHiddenStates []ag.Node,
	pastProjKeysValues multiheadattention.KeysValuesPairs,
) ([]ag.Node, multiheadattention.KeysValuesPairs) {
	residual := m.copy(xs)
	if m.Config.NormalizeBefore {
		xs = m.EncoderAttentionLayerNorm.Forward(xs...)
	}

	qkv := attention.QKV{Queries: xs}
	// use the past key-values if they are available otherwise use the encoder hidden states
	if pastProjKeysValues == nil {
		qkv.Keys = encoderHiddenStates
		qkv.Values = encoderHiddenStates
	}

	att := m.EncoderAttention.ForwardWithPastKeysValues(qkv, pastProjKeysValues)
	xs = att.AttOutput
	// TODO: xs = m.Dropout(xs)
	xs = m.add(residual, xs)
	if !m.Config.NormalizeBefore {
		xs = m.EncoderAttentionLayerNorm.Forward(xs...)
	}
	return xs, att.ProjKeysValues
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
