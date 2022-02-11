// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils"
)

// CalculatePerplexity returns the perplexity for the text calculated as Exp(CrossEntropyLoss).
// The output of the language model is directly compared to the expected targets extracted from the input itself.
func CalculatePerplexity[T mat.DType](m *Model[T], text string) T {
	g := ag.NewGraph[T]()
	defer g.Clear()
	proc := nn.ReifyForInference(m, g)
	sequence := utils.SplitByRune(text)
	prediction := proc.Forward(sequence).([]ag.Node[T])
	targets := targetsIds(sequence, m.Vocabulary, m.UnknownToken)
	loss := losses.CrossEntropySeq(g, prediction[:len(targets)], targets, true)
	return g.Exp(loss).ScalarValue() // perplexity
}
