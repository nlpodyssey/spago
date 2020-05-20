// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
)

// CalculatePerplexity returns the perplexity for the text calculated as Exp(CrossEntropyLoss).
// The output of the language model is directly compared to the expected targets extracted from the input itself.
func CalculatePerplexity(m *Model, text string) float64 {
	sequence := splitByRune(text)
	targetsIds := make([]int, len(sequence)-1) // skip last character
	for i, target := range sequence[1:] {      // the target is always the next character
		id, ok := m.Vocabulary.Id(target)
		if !ok {
			targetsIds[i] = m.Vocabulary.MustId(m.UnknownToken)
			continue
		}
		targetsIds[i] = id
	}
	g := ag.NewGraph()
	proc := m.NewProc(g).(*Processor)
	prediction := proc.Predict(sequence...)[:len(targetsIds)]
	loss := losses.CrossEntropySeq(g, prediction, targetsIds, true)
	return g.Exp(loss).ScalarValue() // perplexity
}
