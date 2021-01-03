// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/utils"
)

// CalculatePerplexity returns the perplexity for the text calculated as Exp(CrossEntropyLoss).
// The output of the language model is directly compared to the expected targets extracted from the input itself.
func CalculatePerplexity(m *Model, text string) mat.Float {
	g := ag.NewGraph()
	defer g.Clear()
	proc := nn.Reify(nn.Context{Graph: g, Mode: nn.Inference}, m).(*Model)
	sequence := utils.SplitByRune(text)
	prediction := proc.Forward(sequence).([]ag.Node)
	targets := targetsIds(sequence, m.Vocabulary, m.UnknownToken)
	loss := losses.CrossEntropySeq(g, prediction[:len(targets)], targets, true)
	return g.Exp(loss).ScalarValue() // perplexity
}
