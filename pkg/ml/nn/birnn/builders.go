// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package birnn

import (
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/cfn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/gru"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/lstm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/ltm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/mist"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/ran"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

// NewBiLSTM returns a new Bidirectional LSTM Model.
func NewBiLSTM(input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  lstm.New(input, hidden),
		Negative:  lstm.New(input, hidden),
		MergeMode: merge,
	}
}

// NewBiGRU returns a new Bidirectional GRU Model.
func NewBiGRU(input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  gru.New(input, hidden),
		Negative:  gru.New(input, hidden),
		MergeMode: merge,
	}
}

// NewBiRAN returns a new Bidirectional RAN Model.
func NewBiRAN(input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  ran.New(input, hidden),
		Negative:  ran.New(input, hidden),
		MergeMode: merge,
	}
}

// NewBiCFN returns a new Bidirectional CFN Model.
func NewBiCFN(input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  cfn.New(input, hidden),
		Negative:  cfn.New(input, hidden),
		MergeMode: merge,
	}
}

// NewBiLTM returns a new Bidirectional LTM Model.
func NewBiLTM(input int, merge MergeType) *Model {
	return &Model{
		Positive:  ltm.New(input),
		Negative:  ltm.New(input),
		MergeMode: merge,
	}
}

// NewBiMIST returns a new Bidirectional MIST Model.
func NewBiMIST(input, hidden, numberOfDelays int, merge MergeType) *Model {
	return &Model{
		Positive:  mist.New(input, hidden, numberOfDelays),
		Negative:  mist.New(input, hidden, numberOfDelays),
		MergeMode: merge,
	}
}

// NewBiBiLSTM returns a new Bidirectional BiLSTM Model.
func NewBiBiLSTM(input, hidden int, merge MergeType) *stack.Model {
	return stack.New(
		NewBiLSTM(input, hidden, Concat),
		NewBiLSTM(hidden*2, hidden, merge),
	)
}
