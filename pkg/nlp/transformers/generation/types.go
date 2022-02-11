// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

package generation

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

// EncoderDecoder is a model able to perform encoder-decoder conditional generation.
type EncoderDecoder[T mat.DType] interface {
	Encoder[T]
	Decoder[T]
	Graph() *ag.Graph[T]
}

// Encoder is a model able to encode each input of a sequence into a vector representation.
type Encoder[T mat.DType] interface {
	// Encode transforms a sequence of input IDs in a sequence of nodes.
	Encode(InputIDs []int) []ag.Node[T]
}

// Decoder is a model able to encode.
type Decoder[T mat.DType] interface {
	// Decode returns the log probabilities for each possible next element of a sequence.
	Decode(encodedInput []ag.Node[T], decodingInputIDs []int, pastCache Cache) (ag.Node[T], Cache)
}

// Scores is just an alias of a Matrix
type Scores[T mat.DType] interface {
	mat.Matrix[T]
}

// Cache is just an alias of interface{}
type Cache interface{}
