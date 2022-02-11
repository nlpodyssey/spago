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
	"github.com/nlpodyssey/spago/pkg/mat"
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
	_ nn.Model[float32] = &Model[float32]{}
)

// Model implements a sequence labeling model.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config          Config
	EmbeddingsLayer *stackedembeddings.Model[T]
	TaggerLayer     *birnncrf.Model[T]
	Labels          []string
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// NewDefaultModel returns a new sequence labeler built based on the architecture of Flair.
// See https://github.com/flairNLP/flair for more information.
func NewDefaultModel[T mat.DType](config Config, path string, readOnlyEmbeddings bool, forceNewEmbeddingsDB bool) *Model[T] {
	CharLanguageModelConfig := charlm.Config{
		VocabularySize:    config.ContextualStringEmbeddings.VocabularySize,
		EmbeddingSize:     config.ContextualStringEmbeddings.EmbeddingSize,
		HiddenSize:        config.ContextualStringEmbeddings.HiddenSize,
		OutputSize:        config.ContextualStringEmbeddings.OutputSize,
		SequenceSeparator: config.ContextualStringEmbeddings.SequenceSeparator,
		UnknownToken:      config.ContextualStringEmbeddings.UnknownToken,
	}

	wordLevelEmbeddings := make([]stackedembeddings.WordsEncoderProcessor[T], len(config.WordEmbeddings))

	for i, weConfig := range config.WordEmbeddings {
		wordLevelEmbeddings[i] = embeddings.New[T](embeddings.Config{
			Size:             weConfig.WordEmbeddingsSize,
			UseZeroEmbedding: true,
			DBPath:           filepath.Join(path, weConfig.WordEmbeddingsFilename),
			ReadOnly:         readOnlyEmbeddings,
			ForceNewDB:       forceNewEmbeddingsDB,
		})
	}

	return &Model[T]{
		Config: config,
		EmbeddingsLayer: &stackedembeddings.Model[T]{
			WordsEncoders: append(
				wordLevelEmbeddings,
				contextualstringembeddings.New[T](
					charlm.New[T](CharLanguageModelConfig),
					charlm.New[T](CharLanguageModelConfig),
					contextualstringembeddings.Concat,
					'\n',
					' ',
				),
			),
			ProjectionLayer: linear.New[T](config.EmbeddingsProjectionInputSize, config.EmbeddingsProjectionOutputSize),
		},
		TaggerLayer: birnncrf.New[T](
			birnn.New[T](
				lstm.New[T](config.RecurrentInputSize, config.RecurrentOutputSize),
				lstm.New[T](config.RecurrentInputSize, config.RecurrentOutputSize),
				birnn.Concat,
			),
			linear.New[T](config.ScorerInputSize, config.ScorerOutputSize),
			crf.New[T](len(config.Labels)),
		),
		Labels: config.Labels,
	}
}

// LoadModel loads a Model from file.
func LoadModel[T mat.DType](modelPath string) (*Model[T], error) {
	config := LoadConfig(filepath.Join(modelPath, "config.json"))
	model := NewDefaultModel[T](
		config,
		modelPath,
		true,  // read-only embeddings
		false, // don't force new embeddings DB
	)

	file := filepath.Join(modelPath, config.ModelFilename)
	fmt.Printf("Loading model parameters from `%s`... ", file)
	err := utils.DeserializeFromFile(file, model)
	if err != nil {
		return nil, fmt.Errorf("sequencelabeler: error during model deserialization")
	}
	// TODO: find a general solution to set embeddings lost during deserialization
	model.loadEmbeddings(
		config,
		modelPath,
		true,  // read-only embeddings
		false, // don't force new embeddings DB
	)
	fmt.Println("ok")
	return model, nil
}

// loadEmbeddings sets the embeddings into the model.
func (m *Model[T]) loadEmbeddings(config Config, path string, readOnlyEmbeddings bool, forceNewEmbeddingsDB bool) {
	for i, weConfig := range config.WordEmbeddings {
		m.EmbeddingsLayer.WordsEncoders[i] = embeddings.New[T](embeddings.Config{
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

// AnalysisResult contains the result of the analysis process.
type AnalysisResult struct {
	// Tokens is the list of annotated tokens.
	Tokens []Token
}

// Analyze returns a list of annotated tokens.
// The result can be adjusted according to the options of merge entities and filter non-entities,
// respectively to merge into one token the pieces of a single recognized entity (e.g. formed by "B-" and "E-"),
// and to discard all tokens that are not recognized as entities (i.e. tag "O").
func (m *Model[T]) Analyze(text string, mergeEntities bool, filterNotEntities bool) AnalysisResult {
	g := ag.NewGraph(ag.ConcurrentComputations[T](runtime.NumCPU()))
	defer g.Clear()
	proc := nn.ReifyForInference(m, g)
	tokenized := basetokenizer.New().Tokenize(text)
	annotated := proc.Forward(tokenized)
	if mergeEntities {
		annotated = m.mergeEntities(annotated)
	}
	if filterNotEntities {
		annotated = m.filterNotEntities(annotated)
	}
	return AnalysisResult{
		Tokens: annotated,
	}
}

// Forward performs the forward step for each input and returns the result.
func (m *Model[T]) Forward(tokens []tokenizers.StringOffsetsPair) []Token {
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
func (m *Model[T]) NegativeLogLoss(emissionScores []ag.Node[T], targets []int) ag.Node[T] {
	return m.TaggerLayer.NegativeLogLoss(emissionScores, targets)
}

// TODO: make sure that the input label sequence is valid
func (m *Model[T]) mergeEntities(tokens []Token) []Token {
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

func (m *Model[T]) filterNotEntities(tokens []Token) []Token {
	ret := make([]Token, 0)
	for _, token := range tokens {
		if token.Label == "O" { // not an entity
			continue
		}
		ret = append(ret, token)
	}
	return ret
}
