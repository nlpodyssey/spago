// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rae

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/pe"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/perceptron"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"math"
)

func NewDefaultEncoder(inputSize, embeddingSize, maxSequenceLength int) *Encoder {
	hiddenSize := int(math.Round(1.5 * float64(embeddingSize)))
	scalingHidden := embeddingSize - ((embeddingSize - inputSize) / 2)

	return &Encoder{
		ScalingFFN: stack.New(
			perceptron.New(inputSize, scalingHidden, ag.Mish),
			perceptron.New(scalingHidden, embeddingSize, ag.Mish)),
		EncodingFFN: stack.New(
			perceptron.New(2*embeddingSize, hiddenSize, ag.Identity),
			layernorm.New(hiddenSize),
			activation.New(ag.Mish),
			perceptron.New(hiddenSize, embeddingSize, ag.Identity),
			layernorm.New(embeddingSize),
			activation.New(ag.Mish)),
		StepEncoder: pe.New(2*embeddingSize, maxSequenceLength),
	}
}

func NewDefaultDecoder(embeddingSize, outputSize, maxSequenceLength int) *Decoder {
	hiddenSize := int(math.Round(1.5 * float64(embeddingSize)))
	descalingHidden := embeddingSize - ((embeddingSize - outputSize) / 2)

	return &Decoder{
		DecodingFNN1: stack.New(
			perceptron.New(embeddingSize, hiddenSize, ag.Identity),
			layernorm.New(hiddenSize),
			activation.New(ag.Mish),
			perceptron.New(hiddenSize, 2*embeddingSize, ag.Identity)),
		DecodingFFN2: stack.New(
			layernorm.New(embeddingSize),
			activation.New(ag.Mish)),
		DescalingFFN: stack.New(
			perceptron.New(embeddingSize, descalingHidden, ag.Mish),
			perceptron.New(descalingHidden, outputSize, ag.Identity)),
		StepEncoder: pe.New(embeddingSize, maxSequenceLength),
	}
}
