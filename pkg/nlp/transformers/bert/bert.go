// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"github.com/nlpodyssey/spago/pkg/mat"
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
	// DefaultConfigurationFile is the default BERT JSON configuration filename.
	DefaultConfigurationFile = "config.json"
	// DefaultVocabularyFile is the default BERT model's vocabulary filename.
	DefaultVocabularyFile = "vocab.txt"
	// DefaultModelFile is the default BERT spaGO model filename.
	DefaultModelFile = "spago_model.bin"
	// DefaultEmbeddingsStorage is the default directory name for BERT model's embedding storage.
	DefaultEmbeddingsStorage = "embeddings_storage"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Config provides configuration settings for a BERT Model.
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
	Training              bool              `json:"training"` // Custom for spaGO
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// LoadConfig loads a BERT model Config from file.
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

// Model implements a BERT model.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config          Config
	Vocabulary      *vocabulary.Vocabulary
	Embeddings      *Embeddings[T]
	Encoder         *Encoder[T]
	Predictor       *Predictor[T]
	Discriminator   *Discriminator[T] // used by "ELECTRA" training method
	Pooler          *Pooler[T]
	SeqRelationship *linear.Model[T]
	SpanClassifier  *SpanClassifier[T]
	Classifier      *Classifier[T]
}

// NewDefaultBERT returns a new model based on the original BERT architecture.
func NewDefaultBERT[T mat.DType](config Config, embeddingsStoragePath string) *Model[T] {
	return &Model[T]{
		Config:     config,
		Vocabulary: nil,
		Embeddings: NewEmbeddings[T](EmbeddingsConfig{
			Size:                config.HiddenSize,
			OutputSize:          config.HiddenSize,
			MaxPositions:        config.MaxPositionEmbeddings,
			TokenTypes:          config.TypeVocabSize,
			WordsMapFilename:    embeddingsStoragePath,
			WordsMapReadOnly:    !config.Training,
			DeletePreEmbeddings: false,
		}),
		Encoder: NewBertEncoder[T](EncoderConfig{
			Size:                   config.HiddenSize,
			NumOfAttentionHeads:    config.NumAttentionHeads,
			IntermediateSize:       config.IntermediateSize,
			IntermediateActivation: ag.OpGELU,
			NumOfLayers:            config.NumHiddenLayers,
		}),
		Predictor: NewPredictor[T](PredictorConfig{
			InputSize:        config.HiddenSize,
			HiddenSize:       config.HiddenSize,
			OutputSize:       config.VocabSize,
			HiddenActivation: ag.OpGELU,
			OutputActivation: ag.OpIdentity, // implicit Softmax (trained with CrossEntropyLoss)
		}),
		Discriminator: NewDiscriminator[T](DiscriminatorConfig{
			InputSize:        config.HiddenSize,
			HiddenSize:       config.HiddenSize,
			HiddenActivation: ag.OpGELU,
			OutputActivation: ag.OpIdentity, // implicit Sigmoid (trained with BCEWithLogitsLoss)
		}),
		Pooler: NewPooler[T](PoolerConfig{
			InputSize:  config.HiddenSize,
			OutputSize: config.HiddenSize,
		}),
		SeqRelationship: linear.New[T](config.HiddenSize, 2),
		SpanClassifier: NewSpanClassifier[T](SpanClassifierConfig{
			InputSize: config.HiddenSize,
		}),
		Classifier: NewTokenClassifier[T](ClassifierConfig{
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

// LoadModel loads a BERT Model from file.
func LoadModel[T mat.DType](modelPath string) (*Model[T], error) {
	configFilename := path.Join(modelPath, DefaultConfigurationFile)
	vocabFilename := path.Join(modelPath, DefaultVocabularyFile)
	embeddingsFilename := path.Join(modelPath, DefaultEmbeddingsStorage)
	modelFilename := path.Join(modelPath, DefaultModelFile)

	log.Printf("Start loading pre-trained model from \"%s\"\n", modelPath)
	log.Printf("[1/4] Load configuration... ")
	config, err := LoadConfig(configFilename)
	if err != nil {
		return nil, err
	}

	log.Printf("[2/4] Instantiate a new model... ")
	model := NewDefaultBERT[T](config, embeddingsFilename)

	log.Printf("[3/4] Load vocabulary... ")
	vocab, err := vocabulary.NewFromFile(vocabFilename)
	if err != nil {
		return nil, err
	}
	model.Vocabulary = vocab

	log.Printf("[4/4] Load model weights... ")
	err = utils.DeserializeFromFile(modelFilename, model)
	if err != nil {
		return nil, fmt.Errorf("bert: error during model deserialization (%s)", err.Error())
	}
	log.Printf("Done.")

	return model, nil
}

// Encode transforms a string sequence into an encoded representation.
func (m *Model[T]) Encode(tokens []string) []ag.Node[T] {
	tokensEncoding := m.Embeddings.Encode(tokens)
	return m.Encoder.Forward(tokensEncoding...)
}

// PredictMasked performs a masked prediction task. It returns the predictions
// for indices associated to the masked nodes.
func (m *Model[T]) PredictMasked(transformed []ag.Node[T], masked []int) map[int]ag.Node[T] {
	return m.Predictor.PredictMasked(transformed, masked)
}

// Discriminate returns 0 or 1 for each encoded element, where 1 means that
// the word is out of context.
func (m *Model[T]) Discriminate(encoded []ag.Node[T]) []int {
	return m.Discriminator.Discriminate(encoded)
}

// Pool "pools" the model by simply taking the hidden state corresponding to the `[CLS]` token.
func (m *Model[T]) Pool(transformed []ag.Node[T]) ag.Node[T] {
	return nn.ToNode[T](m.Pooler.Forward(transformed[0]))
}

// PredictSeqRelationship predicts if the second sentence in the pair is the
// subsequent sentence in the original document.
func (m *Model[T]) PredictSeqRelationship(pooled ag.Node[T]) ag.Node[T] {
	return nn.ToNode[T](m.SeqRelationship.Forward(pooled))
}

// TokenClassification performs a classification for each element in the sequence.
func (m *Model[T]) TokenClassification(transformed []ag.Node[T]) []ag.Node[T] {
	return m.Classifier.Forward(transformed...)
}

// SequenceClassification performs a single sentence-level classification,
// using the pooled CLS token.
func (m *Model[T]) SequenceClassification(transformed []ag.Node[T]) ag.Node[T] {
	return nn.ToNode[T](m.Classifier.Forward(m.Pooler.Forward(transformed[0])...))
}
