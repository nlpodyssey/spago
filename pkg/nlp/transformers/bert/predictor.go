// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model     = &Predictor{}
	_ nn.Processor = &PredictorProcessor{}
)

type PredictorConfig struct {
	InputSize        int
	HiddenSize       int
	OutputSize       int
	HiddenActivation ag.OpName
	OutputActivation ag.OpName
}

type Predictor struct {
	*stack.Model
}

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

type PredictorProcessor struct {
	*stack.Processor
}

func (m *Predictor) NewProc(ctx nn.Context) nn.Processor {
	return &PredictorProcessor{
		Processor: m.Model.NewProc(ctx).(*stack.Processor),
	}
}

func (p *PredictorProcessor) PredictMasked(encoded []ag.Node, masked []int) map[int]ag.Node {
	predictions := make(map[int]ag.Node)
	for _, id := range masked {
		predictions[id] = p.Forward(encoded[id])[0]
	}
	return predictions
}
