// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package birnn

import (
	"github.com/saientist/spago/pkg/ml/nn/rec/cfn"
	"github.com/saientist/spago/pkg/ml/nn/rec/gru"
	"github.com/saientist/spago/pkg/ml/nn/rec/lstm"
	"github.com/saientist/spago/pkg/ml/nn/rec/ltm"
	"github.com/saientist/spago/pkg/ml/nn/rec/mist"
	"github.com/saientist/spago/pkg/ml/nn/rec/ran"
	"github.com/saientist/spago/pkg/ml/nn/stack"
)

func NewBiLSTM(input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  lstm.New(input, hidden),
		Negative:  lstm.New(input, hidden),
		MergeMode: merge,
	}
}

func NewBiGRU(input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  gru.New(input, hidden),
		Negative:  gru.New(input, hidden),
		MergeMode: merge,
	}
}

func NewBiRAN(input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  ran.New(input, hidden),
		Negative:  ran.New(input, hidden),
		MergeMode: merge,
	}
}

func NewBiCFN(input, hidden int, merge MergeType) *Model {
	return &Model{
		Positive:  cfn.New(input, hidden),
		Negative:  cfn.New(input, hidden),
		MergeMode: merge,
	}
}

func NewBiLTM(input int, merge MergeType) *Model {
	return &Model{
		Positive:  ltm.New(input),
		Negative:  ltm.New(input),
		MergeMode: merge,
	}
}

func NewBiMIST(input, hidden, numberOfDelays int, merge MergeType) *Model {
	return &Model{
		Positive:  mist.New(input, hidden, numberOfDelays),
		Negative:  mist.New(input, hidden, numberOfDelays),
		MergeMode: merge,
	}
}

func NewBiBiLSTM(input, hidden int, merge MergeType) *stack.Model {
	return stack.New(
		NewBiLSTM(input, hidden, Concat),
		NewBiLSTM(hidden*2, hidden, merge),
	)
}
