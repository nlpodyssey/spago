// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package barthead

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
	"path"
)

var (
	_ nn.Model     = &SequenceClassification{}
	_ nn.Processor = &SequenceClassificationProcessor{}
)

type SequenceClassification struct {
	BART           *bart.Model
	Classification *Classification
}

func NewSequenceClassification(config bartconfig.Config, embeddingsPath string) *SequenceClassification {
	return &SequenceClassification{
		BART: bart.New(config, embeddingsPath),
		Classification: NewClassification(ClassificationConfig{
			InputSize:     config.DModel,
			HiddenSize:    config.DModel,
			OutputSize:    config.NumLabels,
			PoolerDropout: config.ClassifierDropout,
		}),
	}
}

func (m *SequenceClassification) Close() {
	m.BART.Close()
}

func LoadModelForSequenceClassification(modelPath string) (*SequenceClassification, error) {
	configFilename := path.Join(modelPath, bartconfig.DefaultConfigurationFile)
	embeddingsPath := path.Join(modelPath, bartconfig.DefaultEmbeddingsStorage)
	modelFilename := path.Join(modelPath, bartconfig.DefaultModelFile)

	fmt.Printf("Start loading pre-trained model from \"%s\"\n", modelPath)
	fmt.Printf("[1/2] Loading configuration... ")
	config, err := bartconfig.Load(configFilename)
	if err != nil {
		return nil, err
	}
	fmt.Printf("ok\n")
	model := NewSequenceClassification(config, embeddingsPath)

	fmt.Printf("[2/2] Loading model weights... ")
	err = utils.DeserializeFromFile(modelFilename, nn.NewParamsSerializer(model))
	if err != nil {
		log.Fatal(fmt.Sprintf("bert: error during model deserialization (%s)", err.Error()))
	}
	fmt.Println("ok")

	return model, nil
}

type SequenceClassificationProcessor struct {
	nn.BaseProcessor
	BART           *bart.Processor
	Classification *ClassificationProcessor
}

func (m *SequenceClassification) NewProc(ctx nn.Context) nn.Processor {
	return &SequenceClassificationProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		BART:           m.BART.NewProc(ctx).(*bart.Processor),
		Classification: m.Classification.NewProc(ctx).(*ClassificationProcessor),
	}
}

func (p SequenceClassificationProcessor) Predict(inputIds ...int) []ag.Node {
	transformed := p.BART.Process(inputIds...)
	sentenceRepresentation := transformed[len(transformed)-1]
	return p.Classification.Forward(sentenceRepresentation)
}

func (p SequenceClassificationProcessor) Forward(_ ...ag.Node) []ag.Node {
	panic("barthead: Forward() not implemented for SequenceClassification. Use Predict() instead.")
}
