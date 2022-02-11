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
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model[float32] = &Pooler[float32]{}
)

// PoolerConfig provides configuration settings for a BERT Pooler.
type PoolerConfig struct {
	InputSize  int
	OutputSize int
}

// Pooler is a BERT Pooler model.
type Pooler[T mat.DType] struct {
	*stack.Model[T]
}

func init() {
	gob.Register(&Pooler[float32]{})
	gob.Register(&Pooler[float64]{})
}

// NewPooler returns a new BERT Pooler model.
func NewPooler[T mat.DType](config PoolerConfig) *Pooler[T] {
	return &Pooler[T]{
		Model: stack.New[T](
			linear.New[T](config.InputSize, config.OutputSize),
			activation.New[T](ag.OpTanh),
		),
	}
}
