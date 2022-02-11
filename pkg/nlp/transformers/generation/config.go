// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

package generation

import "github.com/nlpodyssey/spago/pkg/mat"

// GeneratorConfig provides configuration options for the generation search algorithm.
type GeneratorConfig[T mat.DType] struct {
	// NumBeams is the number of beams for generation search.
	NumBeams int
	// MinLength is the minimum length of the sequence to be generated.
	MinLength int
	// MaxLength is the maximum length of the sequence to be generated.
	MaxLength int
	// IsEncoderDecoder reports whether the model is used as an encoder/decoder.
	IsEncoderDecoder bool
	// BOSTokenID is the ID of the Beginning-Of-Sequence token.
	BOSTokenID int
	// EOSTokenID is the ID of the End-Of-Sequence token.
	EOSTokenID int
	// PadTokenID is the id of the padding token.
	PadTokenID int
	// VocabSize is the size of the vocabulary.
	VocabSize int
	// DecoderStartTokenID is the ID of the start token for the decoder of an
	// encoder-decoder model.
	DecoderStartTokenID int
	// LengthPenalty is the exponential penalty to the length.
	// 1.0 means no penalty. Set to values < 1.0 in order to encourage the
	// model to generate shorter sequences, to a value > 1.0 in order to
	// encourage the model to produce longer sequences.
	LengthPenalty T
	// EarlyStopping reports whether to stop the generation search when at least
	// NumBeams sentences are finished per batch or not.
	EarlyStopping bool
	// BadWordsIDs is a list of token IDs that are not allowed to be generated.
	BadWordsIDs [][]int
	// MaxConcurrentComputations is the maximum number of concurrent computations
	// handled by the generation search algorithm.
	MaxConcurrentComputations int
	// IncrementalForward indicates the graph usage mode.
	IncrementalForward bool
}
