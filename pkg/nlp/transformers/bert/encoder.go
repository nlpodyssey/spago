// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package bert provides an implementation of BERT model (Bidirectional Encoder
Representations from Transformers).

Reference: "Attention Is All You Need" by Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin (2017)
(http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
*/
package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model[float32] = &Encoder[float32]{}
)

// EncoderConfig provides configuration parameters for BERT Encoder.
// TODO: include and use the dropout hyper-parameter
type EncoderConfig struct {
	Size                   int
	NumOfAttentionHeads    int
	IntermediateSize       int
	IntermediateActivation ag.OpName
	NumOfLayers            int
}

// Encoder is a BERT Encoder model.
type Encoder[T mat.DType] struct {
	EncoderConfig
	*stack.Model[T]
}

func init() {
	gob.Register(&Encoder[float32]{})
	gob.Register(&Encoder[float64]{})
}

// NewBertEncoder returns a new BERT encoder model composed of a stack of N identical BERT encoder layers.
func NewBertEncoder[T mat.DType](config EncoderConfig) *Encoder[T] {
	return &Encoder[T]{
		EncoderConfig: config,
		Model: stack.Make[T](config.NumOfLayers, func(i int) nn.StandardModel[T] {
			return &EncoderLayer[T]{
				MultiHeadAttention: multiheadattention.New[T](
					config.Size,
					config.NumOfAttentionHeads,
					false, // don't use causal mask
				),
				NormAttention: layernorm.New[T](config.Size),
				FFN: stack.New[T](
					linear.New[T](config.Size, config.IntermediateSize),
					activation.New[T](config.IntermediateActivation),
					linear.New[T](config.IntermediateSize, config.Size),
				),
				NormFFN: layernorm.New[T](config.Size),
				Index:   i,
			}
		}),
	}
}

// NewAlbertEncoder returns a new variant of the BERT encoder model.
// In this variant the stack of N identical BERT encoder layers share the same parameters.
func NewAlbertEncoder[T mat.DType](config EncoderConfig) *Encoder[T] {
	sharedLayer := &EncoderLayer[T]{
		MultiHeadAttention: multiheadattention.New[T](
			config.Size,
			config.NumOfAttentionHeads,
			false, // don't use causal mask
		),
		NormAttention: layernorm.New[T](config.Size),
		FFN: stack.New[T](
			linear.New[T](config.Size, config.IntermediateSize),
			activation.New[T](config.IntermediateActivation),
			linear.New[T](config.IntermediateSize, config.Size),
		),
		NormFFN: layernorm.New[T](config.Size),
	}
	return &Encoder[T]{
		EncoderConfig: config,
		Model: stack.Make[T](config.NumOfLayers, func(_ int) nn.StandardModel[T] {
			return sharedLayer
		}),
	}
}
