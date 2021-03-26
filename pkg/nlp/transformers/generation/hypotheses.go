// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

package generation

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

// Hypotheses provides hypotheses data for a generation Scorer.
type Hypotheses struct {
	config     GeneratorConfig
	beams      []Hypothesis
	worstScore mat.Float
}

// Hypothesis represents a single generation hypothesis, which is a sequence of
// Token IDs paired with a score.
type Hypothesis struct {
	TokenIDs []int
	Score    mat.Float
}

const defaultHypothesisWorstScore mat.Float = 1e9

// NewHypotheses returns a new Hypotheses.
func NewHypotheses(config GeneratorConfig) *Hypotheses {
	return &Hypotheses{
		config:     config,
		beams:      make([]Hypothesis, 0),
		worstScore: 1e9,
	}
}

// Len returns the number of hypotheses in the list.
func (h *Hypotheses) Len() int {
	return len(h.beams)
}

// Add adds a new hypothesis to the list.
func (h *Hypotheses) Add(hypVector []int, sumLogProbs mat.Float) {
	score := sumLogProbs / mat.Pow(mat.Float(len(hypVector)), h.config.LengthPenalty)
	if h.Len() == h.config.NumBeams && score <= h.worstScore {
		return
	}

	h.beams = append(h.beams, Hypothesis{TokenIDs: hypVector, Score: score})
	if h.Len() <= h.config.NumBeams {
		if score < h.worstScore {
			h.worstScore = score
		}
		return
	}

	_, worstIndex, _ := h.findWorst()
	h.beams = append(h.beams[:worstIndex], h.beams[worstIndex+1:]...)

	h.worstScore, _, _ = h.findWorst()
}

func (h *Hypotheses) findWorst() (worstScore mat.Float, worstIndex int, ok bool) {
	if h.Len() == 0 {
		return defaultHypothesisWorstScore, -1, false
	}

	worstIndex = 0
	worstScore = h.beams[0].Score

	for i, hyp := range h.beams[1:] {
		if hyp.Score < worstScore {
			worstIndex = i
			worstScore = hyp.Score
		}
	}

	return worstScore, worstIndex, true
}

// IsDone reports whether there are enough hypotheses and none of the hypotheses
// being generated can become better than the worst one in the heap.
func (h *Hypotheses) IsDone(bestSumLogProbs mat.Float, curLen int) bool {
	if h.Len() < h.config.NumBeams {
		return false
	}
	if h.config.EarlyStopping {
		return true
	}
	curScore := bestSumLogProbs / mat.Pow(mat.Float(curLen), h.config.LengthPenalty)
	return h.worstScore >= curScore
}

// Beams returns the hypothesis beams.
func (h *Hypotheses) Beams() []Hypothesis {
	return h.beams
}
