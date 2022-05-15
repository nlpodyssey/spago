// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package birnn

import (
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn/recurrent/cfn"
	"github.com/nlpodyssey/spago/nn/recurrent/gru"
	"github.com/nlpodyssey/spago/nn/recurrent/lstm"
	"github.com/nlpodyssey/spago/nn/recurrent/ltm"
	"github.com/nlpodyssey/spago/nn/recurrent/mist"
	"github.com/nlpodyssey/spago/nn/recurrent/ran"
	"github.com/nlpodyssey/spago/nn/stack"
)

// NewBiLSTM returns a new Bidirectional LSTM Model.
func NewBiLSTM[T mat.DType](input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  lstm.New[T](input, hidden),
		Negative:  lstm.New[T](input, hidden),
		MergeMode: merge,
	}
}

// NewBiGRU returns a new Bidirectional GRU Model.
func NewBiGRU[T mat.DType](input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  gru.New[T](input, hidden),
		Negative:  gru.New[T](input, hidden),
		MergeMode: merge,
	}
}

// NewBiRAN returns a new Bidirectional RAN Model.
func NewBiRAN[T mat.DType](input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  ran.New[T](input, hidden),
		Negative:  ran.New[T](input, hidden),
		MergeMode: merge,
	}
}

// NewBiCFN returns a new Bidirectional CFN Model.
func NewBiCFN[T mat.DType](input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  cfn.New[T](input, hidden),
		Negative:  cfn.New[T](input, hidden),
		MergeMode: merge,
	}
}

// NewBiLTM returns a new Bidirectional LTM Model.
func NewBiLTM[T mat.DType](input int, merge MergeType) *Model {
	return &Model{
		Positive:  ltm.New[T](input),
		Negative:  ltm.New[T](input),
		MergeMode: merge,
	}
}

// NewBiMIST returns a new Bidirectional MIST Model.
func NewBiMIST[T mat.DType](input, hidden, numberOfDelays int, merge MergeType) *Model {
	return &Model{
		Positive:  mist.New[T](input, hidden, numberOfDelays),
		Negative:  mist.New[T](input, hidden, numberOfDelays),
		MergeMode: merge,
	}
}

// NewBiBiLSTM returns a new Bidirectional BiLSTM Model.
func NewBiBiLSTM[T mat.DType](input, hidden int, merge MergeType) *stack.Model {
	return stack.New(
		NewBiLSTM[T](input, hidden, Concat),
		NewBiLSTM[T](hidden*2, hidden, merge),
	)
}
