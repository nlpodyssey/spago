// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is the first attempt to launch a sequence labeling server from the command line.
// Please note that configurations, parameter loading, and who knows how many other things, require heavy refactoring!
package main

import (
	"archive/tar"
	"compress/gzip"
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
	"github.com/nlpodyssey/spago/pkg/utils/httputils"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
)

var predefinedModels = map[string]string{
	"goflair-en-ner-conll03":      "https://dl.dropboxusercontent.com/s/jgyv568v0nd4ogx/goflair-en-ner-conll03.tar.gz?dl=0",
	"goflair-en-ner-fast-conll03": "https://dl.dropboxusercontent.com/s/9lhh9uom6vh66pg/goflair-en-ner-fast-conll03.tar.gz?dl=0",
}

func main() {
	if len(os.Args) != 4 {
		fmt.Println("Usage:", os.Args[0], "port", "path/to/models/", "model-name")
		return
	}

	port, err := strconv.Atoi(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	modelsFolder := os.Args[2]
	if _, err := os.Stat(modelsFolder); os.IsNotExist(err) {
		log.Fatal(err)
	}

	modelName := os.Args[3]
	modelPath := filepath.Join(modelsFolder, modelName)
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		switch url, ok := predefinedModels[modelName]; {
		case ok:
			fmt.Printf("Fetch model from `%s`\n", url)
			if err := httputils.DownloadFile(fmt.Sprintf("%s-compressed", modelPath), url); err != nil {
				log.Fatal(err)
			}
			r, err := os.Open(fmt.Sprintf("%s-compressed", modelPath))
			if err != nil {
				log.Fatal(err)
			}
			fmt.Print("Extracting compressed model... ")
			ExtractTarGz(r, modelsFolder)
			fmt.Println("ok")
		default:
			log.Fatal(err)
		}
	}

	configPath := filepath.Join(modelPath, "config.json")
	config := loadConfig(configPath)
	model := buildNewDefaultModel(config, modelPath)
	loadModelParams(filepath.Join(modelPath, config.ModelFilename), model)

	fmt.Println(fmt.Sprintf("Start server on port %d.", port))
	server := sequencelabeler.NewServer(model, port)
	server.Start()
}

type Config struct {
	ModelFilename                  string                     `json:"model_filename"`
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
	VocabularySize     int    `json:"vocabulary_size"`
	EmbeddingSize      int    `json:"embedding_size"`
	HiddenSize         int    `json:"hidden_size"`
	SequenceSeparator  string `json:"sequence_separator"`
	UnknownToken       string `json:"unknown_token"`
	VocabularyFilename string `json:"vocabulary_filename"`
}

type WordEmbeddingsConfig struct {
	WordEmbeddingsFilename string `json:"embeddings_filename"`
	WordEmbeddingsSize     int    `json:"embeddings_size"`
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

// ====
// What follows is a copy-and-paste from https://stackoverflow.com/questions/57639648/how-to-decompress-tar-gz-file-in-go
// ====

func ExtractTarGz(gzipStream io.Reader, path string) {
	uncompressedStream, err := gzip.NewReader(gzipStream)
	if err != nil {
		log.Fatal("ExtractTarGz: NewReader failed")
	}
	tarReader := tar.NewReader(uncompressedStream)

	for true {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("ExtractTarGz: Next() failed: %s", err.Error())
		}
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.Mkdir(filepath.Join(path, header.Name), 0755); err != nil {
				log.Fatalf("ExtractTarGz: Mkdir() failed: %s", err.Error())
			}
		case tar.TypeReg:
			outFile, err := os.Create(filepath.Join(path, header.Name))
			if err != nil {
				log.Fatalf("ExtractTarGz: Create() failed: %s", err.Error())
			}
			if _, err := io.Copy(outFile, tarReader); err != nil {
				log.Fatalf("ExtractTarGz: Copy() failed: %s", err.Error())
			}
			outFile.Close()

		default:
			log.Fatalf(
				"ExtractTarGz: uknown type: %d in %s",
				header.Typeflag,
				header.Name)
		}
	}
}
