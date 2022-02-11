// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conditionalgeneration

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/decoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/generation"
	"runtime"
)

var (
	_ nn.Model[float32]                  = &Model[float32]{}
	_ generation.EncoderDecoder[float32] = &Model[float32]{}
)

// Model is a model for conditional generation tasks
// which embeds a BART pre-trained model.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	BART       *bart.Model[T]
	Projection *linear.Model[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new Model for conditional generation.
func New[T mat.DType](config config.Config[T], embeddingsPath string) *Model[T] {
	return &Model[T]{
		BART:       bart.New[T](config, embeddingsPath),
		Projection: linear.New[T](config.DModel, config.VocabSize),
	}
}

// Close closes the BART model's embeddings DB.
func (m *Model[_]) Close() {
	m.BART.Close()
}

// PredictNext returns the logits for the next possible tokens.
func (m *Model[T]) PredictNext(
	encoderOutLastHiddenState []ag.Node[T],
	decoderInputIDs []int,
	pastKeyValues decoder.KeysValuesPairs[T],
) (ag.Node[T], decoder.KeysValuesPairs[T]) {
	decoded, nextCache := m.BART.Decode(decoderInputIDs, encoderOutLastHiddenState, pastKeyValues)
	logits := m.Projection.Forward(decoded...)
	return nn.ToNode[T](logits), nextCache
}

// Generate generates sequences using generation-search decoding.
func (m *Model[T]) Generate(inputIDs []int) []int {
	incrementalForward := m.Graph().IncrementalForwardEnabled()

	maxConcurrentComputations := runtime.NumCPU()
	if incrementalForward {
		maxConcurrentComputations = runtime.NumCPU() / 2
	}

	generator := generation.NewGenerator[T](generation.GeneratorConfig[T]{
		NumBeams:                  m.BART.Config.NumBeams,
		MinLength:                 0,
		MaxLength:                 m.BART.Config.MaxLength,
		IsEncoderDecoder:          m.BART.Config.IsEncoderDecoder,
		BOSTokenID:                m.BART.Config.BosTokenID,
		EOSTokenID:                m.BART.Config.EosTokenID,
		PadTokenID:                m.BART.Config.PadTokenID,
		VocabSize:                 m.BART.Config.VocabSize,
		DecoderStartTokenID:       m.BART.Config.DecoderStartTokenID,
		LengthPenalty:             1.0,
		EarlyStopping:             false,
		BadWordsIDs:               m.BART.Config.BadWordsIDs,
		MaxConcurrentComputations: maxConcurrentComputations,
		IncrementalForward:        incrementalForward,
	}, m)

	return generator.Generate(inputIDs)
}

// Encode satisfies pkg/nlp/transformers/generation/Encoder.
func (m *Model[T]) Encode(InputIDs []int) []ag.Node[T] {
	return m.BART.Encode(InputIDs)
}

// Decode satisfies pkg/nlp/transformers/generation/Decoder.
func (m *Model[T]) Decode(encodedInput []ag.Node[T], inputIDs []int, pastCache generation.Cache) (ag.Node[T], generation.Cache) {
	pastKeysValues, _ := pastCache.(decoder.KeysValuesPairs[T])
	if pastKeysValues != nil {
		// cut input ids if past is used
		inputIDs = inputIDs[len(inputIDs)-1:]
	}
	logits, nextCache := m.PredictNext(encodedInput, inputIDs, pastKeysValues)
	return logits, nextCache
}
