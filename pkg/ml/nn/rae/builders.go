// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package rae provides an implementation of the recursive auto-encoder strategy described in
"Towards Lossless Encoding of Sentences" by Prato et al., 2019.
Unlike the method described in the paper above, here I opted to use the positional encoding
introduced byVaswani et al. (2017) for the step encoding.
*/
package rae

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/pe"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"math"
)

func NewDefaultEncoder(inputSize, embeddingSize, maxSequenceLength int) *Encoder {
	hiddenSize := int(math.Round(1.5 * float64(embeddingSize)))
	scalingHidden := embeddingSize - ((embeddingSize - inputSize) / 2)

	return &Encoder{
		ScalingFFN: stack.New(
			linear.New(inputSize, scalingHidden),
			activation.New(ag.OpMish),
			linear.New(scalingHidden, embeddingSize),
			activation.New(ag.OpMish)),
		EncodingFFN: stack.New(
			linear.New(2*embeddingSize, hiddenSize),
			layernorm.New(hiddenSize),
			activation.New(ag.OpMish),
			linear.New(hiddenSize, embeddingSize),
			layernorm.New(embeddingSize),
			activation.New(ag.OpMish)),
		StepEncoder: pe.NewPositionalEncoder(2*embeddingSize, maxSequenceLength),
	}
}

func NewDefaultDecoder(embeddingSize, outputSize, maxSequenceLength int) *Decoder {
	hiddenSize := int(math.Round(1.5 * float64(embeddingSize)))
	descalingHidden := embeddingSize - ((embeddingSize - outputSize) / 2)

	return &Decoder{
		DecodingFNN1: stack.New(
			linear.New(embeddingSize, hiddenSize),
			layernorm.New(hiddenSize),
			activation.New(ag.OpMish),
			linear.New(hiddenSize, 2*embeddingSize)),
		DecodingFFN2: stack.New(
			layernorm.New(embeddingSize),
			activation.New(ag.OpMish)),
		DescalingFFN: stack.New(
			linear.New(embeddingSize, descalingHidden),
			activation.New(ag.OpMish),
			linear.New(descalingHidden, outputSize)),
		StepEncoder: pe.NewPositionalEncoder(embeddingSize, maxSequenceLength),
	}
}
