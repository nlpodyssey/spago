// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"runtime"
	"sort"
)

// PredictMLM performs the Masked-Language-Model (MLM) prediction.
// It returns the best guess for the masked (i.e. `[MASK]`) tokens in the input text.
func (m *Model) PredictMLM(text string) []Token {
	tokenizer := wordpiecetokenizer.New(m.Vocabulary)
	origTokens := tokenizer.Tokenize(text)
	tokenized := pad(tokenizers.GetStrings(origTokens))

	g := ag.NewGraph(ag.ConcurrentComputations(runtime.NumCPU()))
	defer g.Clear()
	proc := nn.ReifyForInference(m, g).(*Model)
	encoded := proc.Encode(tokenized)

	masked := make([]int, 0)
	for i := range tokenized {
		if tokenized[i] == wordpiecetokenizer.DefaultMaskToken {
			masked = append(masked, i) // target tokens
		}
	}

	retTokens := make(TokenSlice, 0)
	for tokenID, prediction := range proc.PredictMasked(encoded, masked) {
		bestPredictedWordIndex := floatutils.ArgMax(prediction.Value().Data())
		word, ok := m.Vocabulary.Term(bestPredictedWordIndex)
		if !ok {
			word = wordpiecetokenizer.DefaultUnknownToken // if this is returned, there's a misalignment with the vocabulary
		}
		label := DefaultPredictedLabel
		retTokens = append(retTokens, Token{
			Text:  word,
			Start: origTokens[tokenID-1].Offsets.Start, // skip CLS
			End:   origTokens[tokenID-1].Offsets.End,   // skip CLS
			Label: label,
		})
	}

	sort.Sort(retTokens)

	return retTokens
}
