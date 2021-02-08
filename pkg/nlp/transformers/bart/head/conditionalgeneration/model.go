// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conditionalgeneration

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
)

var (
	_ nn.Model = &Model{}
)

// Model is a model for conditional generation tasks
// which embeds a BART pre-trained model.
type Model struct {
	nn.BaseModel
	BART       *bart.Model
	Projection *linear.Model
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Model for conditional generation.
func New(config config.Config, embeddingsPath string) *Model {
	return &Model{
		BART:       bart.New(config, embeddingsPath),
		Projection: linear.New(config.DModel, config.VocabSize),
	}
}

// Close closes the BART model's embeddings DB.
func (m *Model) Close() {
	m.BART.Close()
}
