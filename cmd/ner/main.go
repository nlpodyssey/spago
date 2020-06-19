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
	"io"
	"log"
	"os"
	"os/user"
	"path"
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
	"github.com/nlpodyssey/spago/pkg/utils/httputils"
	"github.com/urfave/cli"
)

const (
	programName = "ner-server"
)

var predefinedModels = map[string]string{
	"goflair-en-ner-conll03":      "https://dl.dropboxusercontent.com/s/jgyv568v0nd4ogx/goflair-en-ner-conll03.tar.gz?dl=0",
	"goflair-en-ner-fast-conll03": "https://dl.dropboxusercontent.com/s/9lhh9uom6vh66pg/goflair-en-ner-fast-conll03.tar.gz?dl=0",
}

func main() {
	app := newNerServerApp()
	app.Run(os.Args)
}

type nerServerApp struct {
	*cli.App
	address      string
	modelsFolder string
	modelName    string
	tlsCert      string
	tlsKey       string
	tlsDisable   bool
}

func newNerServerApp() *nerServerApp {
	app := &nerServerApp{
		App: cli.NewApp(),
	}
	app.Name = programName
	app.Usage = "A demo for named entities recognition."
	app.Commands = []cli.Command{
		newRunCommandFor(app),
	}
	return app
}

func newRunCommandFor(app *nerServerApp) cli.Command {
	return cli.Command{
		Name:        "run",
		Usage:       "Run the " + programName + ".",
		UsageText:   programName + " run --model=<model-name> [--models=<path>] [--address=<address>] [--tls-cert-file=<cert>] [--tls-key-file=<key>] [--tls-disable]",
		Description: "You must indicate the directory that contains the spaGO neural models.",
		Flags:       newRunCommandFlagsFor(app),
		Action:      newRunCommandActionFor(app),
	}
}

func newRunCommandActionFor(app *nerServerApp) func(c *cli.Context) {
	return func(c *cli.Context) {
		fmt.Printf("TLS Cert path is %s\n", app.tlsCert)
		fmt.Printf("TLS private key path is %s\n", app.tlsKey)

		modelsFolder := app.modelsFolder
		if _, err := os.Stat(modelsFolder); os.IsNotExist(err) {
			log.Fatal(err)
		}

		modelName := app.modelName
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

		fmt.Printf("Start %s server listening on %s.\n", func() string {
			if app.tlsDisable {
				return "non-TLS"
			}
			return "TLS"
		}(), app.address)

		server := sequencelabeler.NewServer(model)
		server.Start(app.address, app.tlsCert, app.tlsKey, app.tlsDisable)
	}
}

func newRunCommandFlagsFor(app *nerServerApp) []cli.Flag {
	usr, err := user.Current()
	if err != nil {
		log.Fatal(err)
	}

	return []cli.Flag{
		cli.StringFlag{
			Name:        "address",
			Usage:       "Specifies the bind-address of the server.",
			Value:       "0.0.0.0:1987",
			Destination: &app.address,
		},
		cli.StringFlag{
			Name:        "models",
			Usage:       "Specifies the path to the models.",
			Value:       path.Join(usr.HomeDir, ".spago"),
			Destination: &app.modelsFolder,
		},
		cli.StringFlag{
			Name:        "model-name",
			Usage:       "Specifies the name of the model to use.",
			Destination: &app.modelName,
			Required:    true,
		},
		cli.StringFlag{
			Name:        "tls-cert-file",
			Usage:       "Specifies the path of the TLS certificate file.",
			Value:       "/etc/ssl/certs/spago/server.crt",
			Destination: &app.tlsCert,
		},
		cli.StringFlag{
			Name:        "tls-key-file",
			Usage:       "Specifies the path of the private key for the certificate.",
			Value:       "/etc/ssl/certs/spago/server.key",
			Destination: &app.tlsKey,
		},
		cli.BoolFlag{
			Name:        "tls-disable ",
			Usage:       "Specifies that TLS is disabled.",
			Destination: &app.tlsDisable,
		},
	}
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
