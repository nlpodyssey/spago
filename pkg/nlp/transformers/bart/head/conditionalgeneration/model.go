// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conditionalgeneration

import (
	"encoding/gob"
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
	_ nn.Model                  = &Model{}
	_ generation.EncoderDecoder = &Model{}
)

// Model is a model for conditional generation tasks
// which embeds a BART pre-trained model.
type Model struct {
	nn.BaseModel
	BART       *bart.Model
	Projection *linear.Model
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Model for conditional generation.
func New(config config.Config, embeddingsPath string) *Model {
	return &Model{
		BART:       bart.New(config, embeddingsPath),
		Projection: linear.New(config.DModel, config.VocabSize),
	}
}

// Close closes the BART model's embeddings DB.
func (m *Model) Close() {
	m.BART.Close()
}

// PredictNext returns the logits for the next possible tokens.
func (m *Model) PredictNext(
	encoderOutLastHiddenState []ag.Node,
	decoderInputIDs []int,
	pastKeyValues decoder.KeysValuesPairs,
) (ag.Node, decoder.KeysValuesPairs) {
	decoded, nextCache := m.BART.Decode(decoderInputIDs, encoderOutLastHiddenState, pastKeyValues)
	logits := m.Projection.Forward(decoded...)
	return nn.ToNode(logits), nextCache
}

// Generate generates sequences using generation-search decoding.
func (m *Model) Generate(inputIDs []int) []int {
	incrementalForward := m.Graph().IncrementalForwardEnabled()

	maxConcurrentComputations := 1
	if incrementalForward {
		maxConcurrentComputations = runtime.NumCPU() / 2
	}

	generator := generation.NewGenerator(generation.GeneratorConfig{
		NumBeams:                  m.BART.Config.NumBeams,
		MinLength:                 0,
		MaxLength:                 m.BART.Config.MaxLength,
		IsEncoderDecoder:          m.BART.Config.IsEncoderDecoder,
		BOSTokenID:                m.BART.Config.BosTokenID,
		EOSTokenID:                m.BART.Config.EosTokenID,
		PadTokenID:                m.BART.Config.PadTokenID,
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
func (m *Model) Encode(InputIDs []int) []ag.Node {
	return m.BART.Encode(InputIDs)
}

// Decode satisfies pkg/nlp/transformers/generation/Decoder.
func (m *Model) Decode(encodedInput []ag.Node, inputIDs []int, pastCache generation.Cache) (ag.Node, generation.Cache) {
	pastKeysValues, _ := pastCache.(decoder.KeysValuesPairs)
	if pastKeysValues != nil {
		// cut input ids if past is used
		inputIDs = inputIDs[len(inputIDs)-1:]
	}
	logits, nextCache := m.PredictNext(encodedInput, inputIDs, pastKeysValues)
	logProbs := m.Graph().LogSoftmax(logits)
	return logProbs, nextCache
}
