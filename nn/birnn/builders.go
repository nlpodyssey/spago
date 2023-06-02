// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package birnn

import (
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn/recurrent/gru"
	"github.com/nlpodyssey/spago/nn/recurrent/lstm"
)

// NewBiLSTM returns a new Bidirectional LSTM Model.
func NewBiLSTM[T float.DType](input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  lstm.New[T](input, hidden),
		Negative:  lstm.New[T](input, hidden),
		MergeMode: merge,
	}
}

// NewBiGRU returns a new Bidirectional GRU Model.
func NewBiGRU[T float.DType](input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  gru.New[T](input, hidden),
		Negative:  gru.New[T](input, hidden),
		MergeMode: merge,
	}
}

// NewBiBiLSTM returns a new Bidirectional BiLSTM Model.
func NewBiBiLSTM[T float.DType](input, hidden int, merge MergeType) []*Model {
	return []*Model{
		NewBiLSTM[T](input, hidden, Concat),
		NewBiLSTM[T](hidden*2, hidden, merge),
	}
}
