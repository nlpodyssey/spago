// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is the first attempt to launch a sequence labeling server from the command line.
// Please note that configurations, parameter loading, and who knows how many other things, require heavy refactoring!
package main

import (
	"encoding/json"
	"fmt"
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
	"log"
	"os"
	"strconv"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Println("Usage:", os.Args[0], "port", "path/to/config.json")
		return
	}
	port, err := strconv.Atoi(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	configPath := os.Args[2]
	config := loadConfig(configPath)
	model := buildNewDefaultModel(config)
	loadModelParams(config.ModelPath, model)

	fmt.Println(fmt.Sprintf("Start server on port %d.", port))
	server := sequencelabeler.NewServer(model, port)
	server.Start()
}

type Config struct {
	ModelPath                      string                     `json:"model_path"`
	WordEmbeddings                 WordEmbeddingsConfig       `json:"word_embeddings"`
	ContextualStringEmbeddings     ContextualEmbeddingsConfig `json:"contextual_string_embeddings"`
	EmbeddingsProjectionInputSize  int                        `json:"embeddings_projection_input_size"`
	EmbeddingsProjectionOutputSize int                        `json:"embeddings_projection_output_size"`
	RecurrentInputSize             int                        `json:"recurrent_input_size"`
	RecurrentOutputSize            int                        `json:"recurrent_output_size"`
	ScorerInputSize                int                        `json:"scorer_input_size"`
	ScorerOutputSize               int                        `json:"scorer_output_size"`
	Labels                         []string                   `json:"labels"`
}

type ContextualEmbeddingsConfig struct {
	VocabularySize    int    `json:"vocabulary_size"`
	EmbeddingSize     int    `json:"embedding_size"`
	HiddenSize        int    `json:"hidden_size"`
	SequenceSeparator string `json:"sequence_separator"`
	UnknownToken      string `json:"unknown_token"`
	VocabularyPath    string `json:"vocabulary_path"`
}

type WordEmbeddingsConfig struct {
	WordEmbeddingsPath string `json:"embeddings_path"`
	WordEmbeddingsSize int    `json:"embeddings_size"`
}

func loadConfig(file string) Config {
	var config Config
	configFile, err := os.Open(file)
	if err != nil {
		log.Fatal(err)
	}
	defer configFile.Close()
	err = json.NewDecoder(configFile).Decode(&config)
	if err != nil {
		log.Fatal(err)
	}
	return config
}

// buildNewDefaultModel returns a new sequence labeler built based on the architecture of Flair.
// See https://github.com/flairNLP/flair for more information.
func buildNewDefaultModel(config Config) *sequencelabeler.Model {
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
					DBPath:           config.WordEmbeddings.WordEmbeddingsPath,
					ReadOnly:         true,
					ForceNewDB:       false,
				}),
				contextualstringembeddings.New(
					charlm.New(CharLanguageModelConfig),
					charlm.New(CharLanguageModelConfig),
					contextualstringembeddings.Concat,
					[]rune(config.ContextualStringEmbeddings.SequenceSeparator)[0],
					[]rune(config.ContextualStringEmbeddings.UnknownToken)[0],
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

	vocab := loadVocabulary(config.ContextualStringEmbeddings.VocabularyPath)
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
