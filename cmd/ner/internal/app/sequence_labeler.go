// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package app

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/birnncrf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/crf"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/rec/lstm"
	"github.com/nlpodyssey/spago/pkg/nlp/charlm"
	"github.com/nlpodyssey/spago/pkg/nlp/contextualstringembeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/sequencelabeler"
	"github.com/nlpodyssey/spago/pkg/nlp/stackedembeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"github.com/nlpodyssey/spago/pkg/utils"
)

// buildNewDefaultModel returns a new sequence labeler built based on the architecture of Flair.
// See https://github.com/flairNLP/flair for more information.
func buildNewDefaultModel(config Config, path string) *sequencelabeler.Model {
	CharLanguageModelConfig := charlm.Config{
		VocabularySize:    config.ContextualStringEmbeddings.VocabularySize,
		EmbeddingSize:     config.ContextualStringEmbeddings.EmbeddingSize,
		HiddenSize:        config.ContextualStringEmbeddings.HiddenSize,
		OutputSize:        config.ContextualStringEmbeddings.VocabularySize,
		SequenceSeparator: config.ContextualStringEmbeddings.SequenceSeparator,
		UnknownToken:      config.ContextualStringEmbeddings.UnknownToken,
	}
	m := &sequencelabeler.Model{
		EmbeddingsLayer: &stackedembeddings.Model{
			WordsEncoders: []nn.Model{
				embeddings.New(embeddings.Config{
					Size:             config.WordEmbeddings.WordEmbeddingsSize,
					UseZeroEmbedding: true,
					DBPath:           filepath.Join(path, config.WordEmbeddings.WordEmbeddingsFilename),
					ReadOnly:         true,
					ForceNewDB:       false,
				}),
				contextualstringembeddings.New(
					charlm.New(CharLanguageModelConfig),
					charlm.New(CharLanguageModelConfig),
					contextualstringembeddings.Concat,
					'\n',
					' ',
				),
			},
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

	vocab := loadVocabulary(filepath.Join(path, config.ContextualStringEmbeddings.VocabularyFilename))
	l2rCharLM := m.EmbeddingsLayer.WordsEncoders[1].(*contextualstringembeddings.Model).LeftToRight
	r2lCharLM := m.EmbeddingsLayer.WordsEncoders[1].(*contextualstringembeddings.Model).RightToLeft
	l2rCharLM.Vocabulary, r2lCharLM.Vocabulary = vocab, vocab
	return m
}

func loadVocabulary(file string) *vocabulary.Vocabulary {
	var terms []string
	configFile, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	defer configFile.Close()
	err = json.NewDecoder(configFile).Decode(&terms)
	if err != nil {
		log.Fatal(err)
	}
	return vocabulary.New(terms)
}

func loadModelParams(file string, model *sequencelabeler.Model) {
	fmt.Printf("Loading model parameters from `%s`... ", file)
	err := utils.DeserializeFromFile(file, nn.NewParamsSerializer(model))
	if err != nil {
		panic("error during model deserialization.")
	}
	fmt.Println("ok")
}
