// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bart implements the transformer model introduced by Mike et al., 2019.
// "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
// https://arxiv.org/abs/1910.13461
package bart

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/decoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/encoder"
	"strconv"
)

var (
	_ nn.Model = &Model{}
)

// Model implements a BART model.
type Model struct {
	nn.BaseModel
	Config     config.Config
	Embeddings *embeddings.Model
	Encoder    *encoder.Model
	Decoder    *decoder.Model
}

func init() {
	gob.Register(&Model{})
}

// New returns a new BART Model.
func New(config config.Config, embeddingsStoragePath string) *Model {
	return &Model{
		Config: config,
		Embeddings: embeddings.New(embeddings.Config{
			Size:       config.DModel,
			DBPath:     embeddingsStoragePath,
			ReadOnly:   !config.Training,
			ForceNewDB: false, // TODO: from config?
		}),
		Encoder: encoder.New(config),
		Decoder: decoder.New(config),
	}
}

// Close closes the BART model's embeddings DB.
func (m *Model) Close() {
	m.Embeddings.Close()
}

// Encode performs the forward step for each input and returns the result.
func (m *Model) Encode(inputIDs []int) []ag.Node {
	encoderInput := m.Embeddings.Encode(intToStringSlice(inputIDs))
	encoderOutput := m.Encoder.Encode(encoderInput)
	decoderInput := m.Embeddings.Encode(intToStringSlice(shiftR(inputIDs, 1)))
	decoderOutput, _ := m.Decoder.Decode(decoder.Input{
		Xs:                  decoderInput,
		EncoderHiddenStates: encoderOutput,
		PastKeysValuesPairs: nil,
	})
	return decoderOutput
}

func intToStringSlice(a []int) []string {
	out := make([]string, len(a))
	for i, num := range a {
		out[i] = strconv.Itoa(num)
	}
	return out
}

func shiftR(a []int, i int) []int {
	x, b := a[:(len(a)-i)], a[(len(a)-i):]
	return append(b, x...)
}
