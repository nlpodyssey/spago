// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sequencelabeler

import (
	"encoding/json"
	"log"
	"os"
)

type Config struct {
	ModelFilename                  string                     `json:"model_filename"`
	WordEmbeddings                 WordEmbeddingsConfig       `json:"word_embeddings"`
	WordEmbeddings2                WordEmbeddingsConfig       `json:"word_embeddings_2"`
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
	VocabularySize     int    `json:"vocabulary_size"`
	EmbeddingSize      int    `json:"embedding_size"`
	HiddenSize         int    `json:"hidden_size"`
	OutputSize         int    `json:"output_size"`
	SequenceSeparator  string `json:"sequence_separator"`
	UnknownToken       string `json:"unknown_token"`
	VocabularyFilename string `json:"vocabulary_filename"`
}

type WordEmbeddingsConfig struct {
	WordEmbeddingsFilename string `json:"embeddings_filename"`
	WordEmbeddingsSize     int    `json:"embeddings_size"`
}

func LoadConfig(file string) Config {
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
