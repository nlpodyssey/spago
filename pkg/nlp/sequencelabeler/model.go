// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sequencelabeler provides an implementation of a sequence labeling
//architecture composed by Embeddings -> BiRNN -> Scorer -> CRF.
package sequencelabeler

import (
	"encoding/json"
	"fmt"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnncrf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/crf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/rec/lstm"
	"github.com/nlpodyssey/spago/pkg/nlp/charlm"
	"github.com/nlpodyssey/spago/pkg/nlp/contextualstringembeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/stackedembeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
	"os"
	"path/filepath"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	Config          Config
	EmbeddingsLayer *stackedembeddings.Model
	TaggerLayer     *birnncrf.Model
	Labels          []string
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

	wordLevelEmbeddings := make([]nn.Model, 0)

	if config.WordEmbeddings.WordEmbeddingsSize > 0 {
		wordLevelEmbeddings = append(wordLevelEmbeddings,
			embeddings.New(embeddings.Config{
				Size:             config.WordEmbeddings.WordEmbeddingsSize,
				UseZeroEmbedding: true,
				DBPath:           filepath.Join(path, config.WordEmbeddings.WordEmbeddingsFilename),
				ReadOnly:         readOnlyEmbeddings,
				ForceNewDB:       forceNewEmbeddingsDB,
			}))
	}

	if config.WordEmbeddings2.WordEmbeddingsSize > 0 {
		wordLevelEmbeddings = append(wordLevelEmbeddings,
			embeddings.New(embeddings.Config{
				Size:             config.WordEmbeddings2.WordEmbeddingsSize,
				UseZeroEmbedding: true,
				DBPath:           filepath.Join(path, config.WordEmbeddings2.WordEmbeddingsFilename),
				ReadOnly:         readOnlyEmbeddings,
				ForceNewDB:       forceNewEmbeddingsDB,
			}))
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
		TaggerLayer: &birnncrf.Model{
			BiRNN: birnn.New(
				lstm.New(config.RecurrentInputSize, config.RecurrentOutputSize),
				lstm.New(config.RecurrentInputSize, config.RecurrentOutputSize),
				birnn.Concat,
			),
			Scorer: linear.New(config.ScorerInputSize, config.ScorerOutputSize),
			CRF:    crf.New(len(config.Labels)),
		},
		Labels: config.Labels,
	}
}

func (m *Model) LoadVocabulary(path string) {
	var terms []string
	file, err := os.Open(filepath.Join(path, m.Config.ContextualStringEmbeddings.VocabularyFilename))
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	err = json.NewDecoder(file).Decode(&terms)
	if err != nil {
		log.Fatal(err)
	}

	charLMIndex := 0
	if m.Config.WordEmbeddings.WordEmbeddingsSize > 0 {
		charLMIndex++
	}
	if m.Config.WordEmbeddings2.WordEmbeddingsSize > 0 {
		charLMIndex++
	}
	l2rCharLM := m.EmbeddingsLayer.WordsEncoders[charLMIndex].(*contextualstringembeddings.Model).LeftToRight
	r2lCharLM := m.EmbeddingsLayer.WordsEncoders[charLMIndex].(*contextualstringembeddings.Model).RightToLeft

	vocab := vocabulary.New(terms)
	l2rCharLM.Vocabulary, r2lCharLM.Vocabulary = vocab, vocab
}

func (m *Model) LoadParams(path string) {
	file := filepath.Join(path, m.Config.ModelFilename)
	fmt.Printf("Loading model parameters from `%s`... ", file)
	err := utils.DeserializeFromFile(file, nn.NewParamsSerializer(m))
	if err != nil {
		panic("error during model deserialization.")
	}
	fmt.Println("ok")
}

func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		EmbeddingsLayer: m.EmbeddingsLayer.NewProc(ctx).(*stackedembeddings.Processor),
		TaggerLayer:     m.TaggerLayer.NewProc(ctx).(*birnncrf.Processor),
	}
}

type Processor struct {
	nn.BaseProcessor
	EmbeddingsLayer *stackedembeddings.Processor
	TaggerLayer     *birnncrf.Processor
}

type TokenLabel struct {
	tokenizers.StringOffsetsPair
	Label string
}

func (p *Processor) Predict(tokens []tokenizers.StringOffsetsPair) []TokenLabel {
	model := p.Model.(*Model)
	words := tokenizers.GetStrings(tokens)
	encodings := p.EmbeddingsLayer.Encode(words)
	prediction := p.TaggerLayer.Predict(encodings)
	result := make([]TokenLabel, len(tokens))
	for i, labelIndex := range prediction {
		result[i] = TokenLabel{
			StringOffsetsPair: tokens[i],
			Label:             model.Labels[labelIndex],
		}
	}
	return result
}

// TODO: it could be more consistent if the targets were the string labels
func (p *Processor) NegativeLogLoss(targets []int) ag.Node {
	return p.TaggerLayer.NegativeLogLoss(targets)
}

func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("sequencetagger: method not implemented. Use Predict() instead.")
}
