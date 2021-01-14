// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model = &Classifier{}
)

// ClassifierConfig provides configuration settings for a BERT Classifier.
type ClassifierConfig struct {
	InputSize int
	Labels    []string
}

// Classifier implements a BERT Classifier.
type Classifier struct {
	Config ClassifierConfig
	*linear.Model
}

func init() {
	gob.Register(&Classifier{})
}

// NewTokenClassifier returns a new BERT Classifier model.
func NewTokenClassifier(config ClassifierConfig) *Classifier {
	return &Classifier{
		Config: config,
		Model:  linear.New(config.InputSize, len(config.Labels)),
	}
}
