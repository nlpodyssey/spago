// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

package generation

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"sort"
)

// Scorer is a generation scorer implementing standard generation search decoding.
type Scorer struct {
	config     GeneratorConfig
	hypotheses *Hypotheses
	isDone     bool
}

// ScoredToken associates a score to a token identified by its
// (generation-index, token-index) position.
type ScoredToken struct {
	BeamIndex  int
	TokenIndex int
	Score      mat.Float
}

// ScoredTokens is a slice of ScoredToken.
type ScoredTokens []*ScoredToken

// ScorerProcessOutput is the output value of Scorer.Process.
type ScorerProcessOutput struct {
	nextBeamScores  []mat.Float
	nextBeamTokens  []int
	nextBeamIndices []int
}

// NewScorer returns a new Scorer.
func NewScorer(config GeneratorConfig) *Scorer {
	return &Scorer{
		config:     config,
		hypotheses: NewHypotheses(config),
		isDone:     false,
	}
}

// IsDone reports whether there are enough hypotheses and none of the hypotheses
// being generated can become better than the worst one in the heap.
func (s *Scorer) IsDone() bool {
	return s.isDone
}

// Process processes a new set of scored tokens.
func (s *Scorer) Process(inputIDs [][]int, scoredTokens ScoredTokens) ScorerProcessOutput {
	numBeams := s.config.NumBeams
	eosTokenID := s.config.EOSTokenID
	padTokenID := s.config.PadTokenID
	curLen := len(inputIDs[0])

	out := ScorerProcessOutput{
		nextBeamScores:  make([]mat.Float, numBeams),
		nextBeamTokens:  make([]int, numBeams),
		nextBeamIndices: make([]int, numBeams),
	}

	if s.isDone {
		for i := range out.nextBeamTokens {
			out.nextBeamTokens[i] = padTokenID
		}
		return out
	}

	// next tokens for this sentence
	beamIdx := 0

	for beamTokenRank, scoredToken := range scoredTokens {
		// add to generated hypotheses if end of sentence
		if eosTokenID >= 0 && scoredToken.TokenIndex == eosTokenID {
			// if the token does not belong to top numBeans tokens, it should not be added
			if beamTokenRank >= numBeams {
				continue
			}
			hypVec := make([]int, len(inputIDs[scoredToken.BeamIndex]))
			copy(hypVec, inputIDs[scoredToken.BeamIndex])
			s.hypotheses.Add(hypVec, scoredToken.Score)
		} else {
			// add next predicted token since it is not eos_token
			out.nextBeamScores[beamIdx] = scoredToken.Score
			out.nextBeamTokens[beamIdx] = scoredToken.TokenIndex
			out.nextBeamIndices[beamIdx] = scoredToken.BeamIndex
			beamIdx++
		}

		// once the generation for next step is full, don't add more tokens to it.
		if beamIdx == numBeams {
			break
		}
	}

	// Check if we are done so that we can save a pad step
	// (note: scoredTokens[0] contains the max score)
	s.isDone = s.hypotheses.IsDone(scoredTokens[0].Score, curLen)

	return out
}

// Finalize finalizes the generation hypotheses and returns the best sequence.
func (s *Scorer) Finalize(inputIDs [][]int, finalBeamScores []mat.Float) []int {
	eosTokenID := s.config.EOSTokenID
	numBeams := s.config.NumBeams

	// Finalize all open generation hypotheses and add to generated hypotheses.
	if !s.isDone {
		// All open generation hypotheses are added to the generation hypothesis.
		// Generator hypothesis class automatically keeps the best beams.
		for beamID := 0; beamID < numBeams; beamID++ {
			finalScore := finalBeamScores[beamID]
			finalTokens := inputIDs[beamID]
			s.hypotheses.Add(finalTokens, finalScore)
		}
	}

	beams := s.hypotheses.Beams()
	sort.Slice(beams, func(i, j int) bool {
		return beams[i].Score > beams[j].Score
	})

	bestTokensSequence := beams[0].TokenIDs
	if len(bestTokensSequence) < s.config.MaxLength {
		bestTokensSequence = append(bestTokensSequence, eosTokenID)
	}
	return bestTokensSequence
}
