// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model     = &Pooler{}
	_ nn.Processor = &PoolerProcessor{}
)

// PoolerConfig provides configuration settings for a BERT Pooler.
type PoolerConfig struct {
	InputSize  int
	OutputSize int
}

// Pooler is a BERT Pooler model.
type Pooler struct {
	*stack.Model
}

// NewPooler returns a new BERT Pooler model.
func NewPooler(config PoolerConfig) *Pooler {
	return &Pooler{
		Model: stack.New(
			linear.New(config.InputSize, config.OutputSize),
			activation.New(ag.OpTanh),
		),
	}
}

// PoolerProcessor implements a nn.Processor for a BERT Pooler.
type PoolerProcessor struct {
	*stack.Processor
}

// NewProc returns a new processor to execute the forward step.
func (m *Pooler) NewProc(ctx nn.Context) nn.Processor {
	return &PoolerProcessor{
		Processor: m.Model.NewProc(ctx).(*stack.Processor),
	}
}
