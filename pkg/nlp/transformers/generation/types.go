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
type EncoderDecoder interface {
	Encoder
	Decoder
	Graph() *ag.Graph
}

// Encoder is a model able to encode each input of a sequence into a vector representation.
type Encoder interface {
	// Encode transforms a sequence of input IDs in a sequence of nodes.
	Encode(InputIDs []int) []ag.Node
}

// Decoder is a model able to encode.
type Decoder interface {
	// Decode returns the log probabilities for each possible next element of a sequence.
	Decode(encodedInput []ag.Node, decodingInputIDs []int, pastCache Cache) (ag.Node, Cache)
}

// Scores is just an alias of a Matrix
type Scores = mat.Matrix[mat.Float]

// Cache is just an alias of interface{}
type Cache interface{}
