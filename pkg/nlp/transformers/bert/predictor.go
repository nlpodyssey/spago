// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model = &Predictor{}
)

// PredictorConfig provides configuration settings for a BERT Predictor.
type PredictorConfig struct {
	InputSize        int
	HiddenSize       int
	OutputSize       int
	HiddenActivation ag.OpName
	OutputActivation ag.OpName
}

// Predictor is a BERT Predictor model.
type Predictor struct {
	*stack.Model
}

func init() {
	gob.Register(&Predictor{})
}

// NewPredictor returns a new BERT Predictor model.
func NewPredictor(config PredictorConfig) *Predictor {
	return &Predictor{
		Model: stack.New(
			linear.New(config.InputSize, config.HiddenSize),
			activation.New(config.HiddenActivation),
			layernorm.New(config.HiddenSize),
			linear.New(config.HiddenSize, config.OutputSize),
			activation.New(config.OutputActivation),
		),
	}
}

// PredictMasked performs a masked prediction task. It returns the predictions
// for indices associated to the masked nodes.
func (m *Predictor) PredictMasked(encoded []ag.Node, masked []int) map[int]ag.Node {
	predictions := make(map[int]ag.Node)
	for _, id := range masked {
		predictions[id] = nn.ToNode(m.Forward(encoded[id]))
	}
	return predictions
}
