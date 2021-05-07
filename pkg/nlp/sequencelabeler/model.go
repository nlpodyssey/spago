// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sequencelabeler provides an implementation of a sequence labeling
// architecture composed by Embeddings -> BiRNN -> Scorer -> CRF.
package sequencelabeler

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnncrf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/crf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/lstm"
	"github.com/nlpodyssey/spago/pkg/nlp/charlm"
	"github.com/nlpodyssey/spago/pkg/nlp/contextualstringembeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/stackedembeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/basetokenizer"
	"github.com/nlpodyssey/spago/pkg/utils"
	"path/filepath"
	"runtime"
)

var (
	_ nn.Model = &Model{}
)

// Model implements a sequence labeling model.
type Model struct {
	nn.BaseModel
	Config          Config
	EmbeddingsLayer *stackedembeddings.Model
	TaggerLayer     *birnncrf.Model
	Labels          []string
}

func init() {
	gob.Register(&Model{})
}

// NewDefaultModel returns a new sequence labeler built based on the architecture of Flair.
// See https://github.com/flairNLP/flair for more information.
func NewDefaultModel(config Config, path string, readOnlyEmbeddings bool, forceNewEmbeddingsDB bool) *Model {
	CharLanguageModelConfig := charlm.Config{
		VocabularySize:    config.ContextualStringEmbeddings.VocabularySize,
		EmbeddingSize:     config.ContextualStringEmbeddings.EmbeddingSize,
		HiddenSize:        config.ContextualStringEmbeddings.HiddenSize,
		OutputSize:        config.ContextualStringEmbeddings.OutputSize,
		SequenceSeparator: config.ContextualStringEmbeddings.SequenceSeparator,
		UnknownToken:      config.ContextualStringEmbeddings.UnknownToken,
	}

	wordLevelEmbeddings := make([]stackedembeddings.WordsEncoderProcessor, len(config.WordEmbeddings))

	for i, weConfig := range config.WordEmbeddings {
		wordLevelEmbeddings[i] = embeddings.New(embeddings.Config{
			Size:             weConfig.WordEmbeddingsSize,
			UseZeroEmbedding: true,
			DBPath:           filepath.Join(path, weConfig.WordEmbeddingsFilename),
			ReadOnly:         readOnlyEmbeddings,
			ForceNewDB:       forceNewEmbeddingsDB,
		})
	}

	return &Model{
		Config: config,
		EmbeddingsLayer: &stackedembeddings.Model{
			WordsEncoders: append(
				wordLevelEmbeddings,
				contextualstringembeddings.New(
					charlm.New(CharLanguageModelConfig),
					charlm.New(CharLanguageModelConfig),
					contextualstringembeddings.Concat,
					'\n',
					' ',
				),
			),
			ProjectionLayer: linear.New(config.EmbeddingsProjectionInputSize, config.EmbeddingsProjectionOutputSize),
		},
		TaggerLayer: birnncrf.New(
			birnn.New(
				lstm.New(config.RecurrentInputSize, config.RecurrentOutputSize),
				lstm.New(config.RecurrentInputSize, config.RecurrentOutputSize),
				birnn.Concat,
			),
			linear.New(config.ScorerInputSize, config.ScorerOutputSize),
			crf.New(len(config.Labels)),
		),
		Labels: config.Labels,
	}
}

// Load loads a Model from file.
func Load(path string) (*Model, error) {
	config := LoadConfig(filepath.Join(path, "config.json"))
	model := NewDefaultModel(
		config,
		path,
		true,  // read-only embeddings
		false, // don't force new embeddings DB
	)

	file := filepath.Join(path, config.ModelFilename)
	fmt.Printf("Loading model parameters from `%s`... ", file)
	err := utils.DeserializeFromFile(file, model)
	if err != nil {
		return nil, fmt.Errorf("sequencelabeler: error during model deserialization")
	}
	// TODO: find a general solution to set embeddings lost during deserialization
	model.loadEmbeddings(
		config,
		path,
		true,  // read-only embeddings
		false, // don't force new embeddings DB
	)
	fmt.Println("ok")
	return model, nil
}

// loadEmbeddings sets the embeddings into the model.
func (m *Model) loadEmbeddings(config Config, path string, readOnlyEmbeddings bool, forceNewEmbeddingsDB bool) {
	for i, weConfig := range config.WordEmbeddings {
		m.EmbeddingsLayer.WordsEncoders[i] = embeddings.New(embeddings.Config{
			Size:             weConfig.WordEmbeddingsSize,
			UseZeroEmbedding: true,
			DBPath:           filepath.Join(path, weConfig.WordEmbeddingsFilename),
			ReadOnly:         readOnlyEmbeddings,
			ForceNewDB:       forceNewEmbeddingsDB,
		})
	}
}

// Token is a token resulting from the analysis process.
type Token struct {
	Text  string `json:"text"`
	Start int    `json:"start"`
	End   int    `json:"end"`
	Label string `json:"label"`
}

// Analyze returns a slice of annotated tokens.
// The result can be adjusted according to the options of merge entities and filter non-entities,
// respectively to merge into one token the pieces of a single recognized entity (e.g. formed by "B-" and "E-"),
// and to discard all tokens that are not recognized as entities (i.e. tag "O").
func (m *Model) Analyze(text string, mergeEntities bool, filterNotEntities bool) []Token {
	g := ag.NewGraph(ag.ConcurrentComputations(runtime.NumCPU()))
	defer g.Clear()
	proc := nn.Reify(nn.Context{Graph: g, Mode: nn.Inference}, m).(*Model)
	tokenized := basetokenizer.New().Tokenize(text)
	result := proc.Forward(tokenized)
	if mergeEntities {
		result = m.mergeEntities(result)
	}
	if filterNotEntities {
		result = m.filterNotEntities(result)
	}
	return result
}

// Forward performs the forward step for each input and returns the result.
func (m *Model) Forward(tokens []tokenizers.StringOffsetsPair) []Token {
	words := tokenizers.GetStrings(tokens)
	encodings := m.EmbeddingsLayer.Encode(words)
	prediction := m.TaggerLayer.Predict(encodings)
	result := make([]Token, len(tokens))
	for i, labelIndex := range prediction {
		tk := tokens[i]
		result[i] = Token{
			Text:  tk.String,
			Start: tk.Offsets.Start,
			End:   tk.Offsets.End,
			Label: m.Labels[labelIndex],
		}
	}
	return result
}

// NegativeLogLoss computes the negative log loss with respect to the targets.
// TODO: it could be more consistent if the targets were the string labels
func (m *Model) NegativeLogLoss(emissionScores []ag.Node, targets []int) ag.Node {
	return m.TaggerLayer.NegativeLogLoss(emissionScores, targets)
}

// TODO: make sure that the input label sequence is valid
func (m *Model) mergeEntities(tokens []Token) []Token {
	newTokens := make([]Token, 0)
	buf := Token{}
	text := bytes.NewBufferString("")
	for _, token := range tokens {
		switch token.Label[0] {
		case 'O':
			newTokens = append(newTokens, token)
		case 'S':
			newToken := token
			newToken.Label = newToken.Label[2:]
			newTokens = append(newTokens, newToken)
		case 'B':
			text.Reset()
			text.Write([]byte(token.Text))
			buf = Token{}
			buf.Label = fmt.Sprintf("%s", token.Label[2:]) // copy
			buf.Start = token.Start
		case 'I':
			text.Write([]byte(fmt.Sprintf(" %s", token.Text)))
		case 'E':
			text.Write([]byte(fmt.Sprintf(" %s", token.Text)))
			buf.Text = text.String()
			buf.End = token.End
			newTokens = append(newTokens, buf)
		}
	}
	return newTokens
}

func (m *Model) filterNotEntities(tokens []Token) []Token {
	ret := make([]Token, 0)
	for _, token := range tokens {
		if token.Label == "O" { // not an entity
			continue
		}
		ret = append(ret, token)
	}
	return ret
}
