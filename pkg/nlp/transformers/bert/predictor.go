// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model[float32] = &Predictor[float32]{}
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
type Predictor[T mat.DType] struct {
	*stack.Model[T]
}

func init() {
	gob.Register(&Predictor[float32]{})
	gob.Register(&Predictor[float64]{})
}

// NewPredictor returns a new BERT Predictor model.
func NewPredictor[T mat.DType](config PredictorConfig) *Predictor[T] {
	return &Predictor[T]{
		Model: stack.New[T](
			linear.New[T](config.InputSize, config.HiddenSize),
			activation.New[T](config.HiddenActivation),
			layernorm.New[T](config.HiddenSize),
			linear.New[T](config.HiddenSize, config.OutputSize),
			activation.New[T](config.OutputActivation),
		),
	}
}

// PredictMasked performs a masked prediction task. It returns the predictions
// for indices associated to the masked nodes.
func (m *Predictor[T]) PredictMasked(encoded []ag.Node[T], masked []int) map[int]ag.Node[T] {
	predictions := make(map[int]ag.Node[T])
	for _, id := range masked {
		predictions[id] = nn.ToNode[T](m.Forward(encoded[id]))
	}
	return predictions
}
