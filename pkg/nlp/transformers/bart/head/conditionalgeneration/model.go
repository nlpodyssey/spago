// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conditionalgeneration

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
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
		MaxConcurrentComputations: runtime.NumCPU() / 2,
	}, m)

	return generator.Generate(inputIDs)
}

// Encode satisfies pkg/nlp/transformers/generation/Encoder.
func (m *Model) Encode(InputIDs []int) []ag.Node {
	return m.BART.Encode(InputIDs)
}

// Encode satisfies pkg/nlp/transformers/generation/Decoder.
func (m *Model) Decode(encodedInput []ag.Node, inputIDs []int, curLen int, pastCache generation.Cache) (generation.Scores, generation.Cache) {
	pastKeysValues, _ := pastCache.(decoder.KeysValuesPairs)
	if pastKeysValues != nil {
		// cut input ids if past is used
		inputIDs = inputIDs[len(inputIDs)-1:]
	}

	logits, nextCache := m.PredictNext(encodedInput, inputIDs, pastKeysValues)

	// important: detach the logits from the current computation branch
	// (so that they can be modified in place without side effects)
	logits = m.Graph().NewVariable(logits.Value(), false)
	m.adjustLogits(logits.Value(), curLen)

	scores := m.logSoftmax(logits)
	return m.Graph().GetCopiedValue(scores), nextCache
}

func (m *Model) logSoftmax(logits ag.Node) ag.Node {
	g := m.Graph()
	return g.Log(g.Softmax(logits))
}

func (m *Model) adjustLogits(logits mat.Matrix, curLen int) mat.Matrix {
	maxLen := m.BART.Config.MaxLength
	padTokenID := m.BART.Config.PadTokenID
	eosTokenID := m.BART.Config.EosTokenID

	// Never predict pad token.
	logits.SetVec(padTokenID, mat.Inf(-1))

	if curLen == maxLen-1 && eosTokenID >= 0 {
		// Force EOS token ID to be generated.
		for i := 0; i < logits.Size(); i++ {
			if i == eosTokenID {
				continue
			}
			logits.SetVec(i, mat.Inf(-1))
		}
	}

	return logits
}
