// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/json"
	"fmt"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
	"os"
	"path"
	"strconv"
)

const (
	DefaultConfigurationFile = "config.json"
	DefaultVocabularyFile    = "vocab.txt"
	DefaultModelFile         = "spago_model.bin"
	DefaultEmbeddingsStorage = "embeddings_storage"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Config struct {
	HiddenAct             string            `json:"hidden_act"`
	HiddenSize            int               `json:"hidden_size"`
	IntermediateSize      int               `json:"intermediate_size"`
	MaxPositionEmbeddings int               `json:"max_position_embeddings"`
	NumAttentionHeads     int               `json:"num_attention_heads"`
	NumHiddenLayers       int               `json:"num_hidden_layers"`
	TypeVocabSize         int               `json:"type_vocab_size"`
	VocabSize             int               `json:"vocab_size"`
	ID2Label              map[string]string `json:"id2label"`
	ReadOnly              bool              `json:"read_only"`
}

func LoadConfig(file string) (Config, error) {
	var config Config
	configFile, err := os.Open(file)
	if err != nil {
		return Config{}, err
	}
	defer configFile.Close()
	err = json.NewDecoder(configFile).Decode(&config)
	if err != nil {
		return Config{}, err
	}
	return config, nil
}

type Model struct {
	Config          Config
	Vocabulary      *vocabulary.Vocabulary
	Embeddings      *Embeddings
	Encoder         *Encoder
	Predictor       *Predictor
	Discriminator   *Discriminator // used by "ELECTRA" training method
	Pooler          *Pooler
	SeqRelationship *linear.Model
	SpanClassifier  *SpanClassifier
	Classifier      *Classifier
}

// NewDefaultBERT returns a new model based on the original BERT architecture.
func NewDefaultBERT(config Config, embeddingsStoragePath string) *Model {
	return &Model{
		Config:     config,
		Vocabulary: nil,
		Embeddings: NewEmbeddings(EmbeddingsConfig{
			Size:                config.HiddenSize,
			OutputSize:          config.HiddenSize,
			MaxPositions:        config.MaxPositionEmbeddings,
			TokenTypes:          config.TypeVocabSize,
			WordsMapFilename:    embeddingsStoragePath,
			WordsMapReadOnly:    config.ReadOnly,
			DeletePreEmbeddings: false,
		}),
		Encoder: NewBertEncoder(EncoderConfig{
			Size:                   config.HiddenSize,
			NumOfAttentionHeads:    config.NumAttentionHeads,
			IntermediateSize:       config.IntermediateSize,
			IntermediateActivation: ag.OpGeLU,
			NumOfLayers:            config.NumHiddenLayers,
		}),
		Predictor: NewPredictor(PredictorConfig{
			InputSize:        config.HiddenSize,
			HiddenSize:       config.HiddenSize,
			OutputSize:       config.VocabSize,
			HiddenActivation: ag.OpGeLU,
			OutputActivation: ag.OpIdentity, // implicit Softmax (trained with CrossEntropyLoss)
		}),
		Discriminator: NewDiscriminator(DiscriminatorConfig{
			InputSize:        config.HiddenSize,
			HiddenSize:       config.HiddenSize,
			HiddenActivation: ag.OpGeLU,
			OutputActivation: ag.OpIdentity, // implicit Sigmoid (trained with BCEWithLogitsLoss)
		}),
		Pooler: NewPooler(PoolerConfig{
			InputSize:  config.HiddenSize,
			OutputSize: config.HiddenSize,
		}),
		SeqRelationship: linear.New(config.HiddenSize, 2),
		SpanClassifier: NewSpanClassifier(SpanClassifierConfig{
			InputSize: config.HiddenSize,
		}),
		Classifier: NewTokenClassifier(ClassifierConfig{
			InputSize: config.HiddenSize,
			Labels: func(x map[string]string) []string {
				if len(x) == 0 {
					return []string{"LABEL_0", "LABEL_1"} // assume binary classification by default
				}
				y := make([]string, len(x))
				for k, v := range x {
					i, err := strconv.Atoi(k)
					if err != nil {
						log.Fatal(err)
					}
					y[i] = v
				}
				return y
			}(config.ID2Label),
		}),
	}
}

func LoadModel(modelPath string) (*Model, error) {
	configFilename := path.Join(modelPath, DefaultConfigurationFile)
	vocabFilename := path.Join(modelPath, DefaultVocabularyFile)
	embeddingsFilename := path.Join(modelPath, DefaultEmbeddingsStorage)
	modelFilename := path.Join(modelPath, DefaultModelFile)

	fmt.Printf("Start loading pre-trained model from \"%s\"\n", modelPath)
	fmt.Printf("[1/3] Loading configuration... ")
	config, err := LoadConfig(configFilename)
	if err != nil {
		return nil, err
	}
	fmt.Printf("ok\n")
	model := NewDefaultBERT(config, embeddingsFilename)

	fmt.Printf("[2/3] Loading vocabulary... ")
	vocab, err := vocabulary.NewFromFile(vocabFilename)
	if err != nil {
		return nil, err
	}
	fmt.Printf("ok\n")
	model.Vocabulary = vocab

	fmt.Printf("[3/3] Loading model weights... ")
	err = utils.DeserializeFromFile(modelFilename, nn.NewParamsSerializer(model))
	if err != nil {
		log.Fatal(fmt.Sprintf("bert: error during model deserialization (%s)", err.Error()))
	}
	fmt.Println("ok")

	return model, nil
}

type Processor struct {
	nn.BaseProcessor
	Embeddings      *EmbeddingsProcessor
	Encoder         *EncoderProcessor
	Predictor       *PredictorProcessor
	Discriminator   *DiscriminatorProcessor
	Pooler          *PoolerProcessor
	SeqRelationship *linear.Processor
	SpanClassifier  *SpanClassifierProcessor
	Classifier      *ClassifierProcessor
}

func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		Embeddings:      m.Embeddings.NewProc(ctx).(*EmbeddingsProcessor),
		Encoder:         m.Encoder.NewProc(ctx).(*EncoderProcessor),
		Predictor:       m.Predictor.NewProc(ctx).(*PredictorProcessor),
		Discriminator:   m.Discriminator.NewProc(ctx).(*DiscriminatorProcessor),
		Pooler:          m.Pooler.NewProc(ctx).(*PoolerProcessor),
		SeqRelationship: m.SeqRelationship.NewProc(ctx).(*linear.Processor),
		SpanClassifier:  m.SpanClassifier.NewProc(ctx).(*SpanClassifierProcessor),
		Classifier:      m.Classifier.NewProc(ctx).(*ClassifierProcessor),
	}
}

func (p *Processor) Encode(tokens []string) []ag.Node {
	tokensEncoding := p.Embeddings.Encode(tokens)
	return p.Encoder.Forward(tokensEncoding...)
}

func (p *Processor) PredictMasked(transformed []ag.Node, masked []int) map[int]ag.Node {
	return p.Predictor.PredictMasked(transformed, masked)
}

func (p *Processor) Discriminate(encoded []ag.Node) []int {
	return p.Discriminator.Discriminate(encoded)
}

// Pool "pools" the model by simply taking the hidden state corresponding to the `[CLS]` token.
func (p *Processor) Pool(transformed []ag.Node) ag.Node {
	return p.Pooler.Forward(transformed[0])[0]
}

func (p *Processor) PredictSeqRelationship(pooled ag.Node) ag.Node {
	return p.SeqRelationship.Forward(pooled)[0]
}

func (p *Processor) TokenClassification(transformed []ag.Node) []ag.Node {
	return p.Classifier.Predict(transformed)
}

func (p *Processor) SequenceClassification(transformed []ag.Node) ag.Node {
	return p.Classifier.Predict(p.Pooler.Forward(transformed[0]))[0]
}

func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("bert: method not implemented")
}
