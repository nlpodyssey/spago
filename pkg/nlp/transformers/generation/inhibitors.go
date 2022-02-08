// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

package generation

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/utils"
)

func (b *Generator) inhibitInvalidTokens(inputIDs [][]int, scores []Scores) []Scores {
	if b.config.MinLength >= 0 && b.config.EOSTokenID >= 0 {
		scores = b.processMinLengthScores(inputIDs, scores)
	}
	if len(b.config.BadWordsIDs) > 0 {
		scores = b.processBadWordsScores(inputIDs, scores)
	}
	return scores
}

func (b *Generator) processBadWordsScores(inputIDs [][]int, scores []Scores) []Scores {
	BadWordsIDs := make([][]int, 0, len(b.config.BadWordsIDs))
	for _, v := range b.config.BadWordsIDs {
		if len(v) == 1 && v[0] == b.config.EOSTokenID {
			continue
		}
		BadWordsIDs = append(BadWordsIDs, v)
	}

	// Calculate banned bad words IDs
	bannedTokens := make([][]int, 0, len(inputIDs))
	for _, slice := range inputIDs {
		bannedTokensSlice := make([]int, 0, len(slice))
		for _, bannedTokenSeq := range BadWordsIDs {
			if !bannedTokensMatch(slice, bannedTokenSeq[:len(bannedTokenSeq)-1]) {
				continue
			}
			bannedTokensSlice = append(bannedTokensSlice, bannedTokenSeq[len(bannedTokenSeq)-1])
		}
		bannedTokens = append(bannedTokens, bannedTokensSlice)
	}

	// Set scores to -Inf for banned tokens
	for idx, batchBannedTokens := range bannedTokens {
		for _, tokenID := range batchBannedTokens {
			scores[idx].SetVec(tokenID, mat.Inf[mat.Float](-1))
		}
	}

	return scores
}

func bannedTokensMatch(prevTokens []int, bannedTokens []int) bool {
	if len(bannedTokens) == 0 {
		// If bad word tokens is just one token always ban it.
		return true
	}
	if len(bannedTokens) > len(prevTokens) {
		// If bad word tokens are longer then prev input_ids they can't be equal.
		return false
	}
	return utils.IntSliceEqual(prevTokens[len(prevTokens)-len(bannedTokens):], bannedTokens)
}

func (b *Generator) processMinLengthScores(inputIDs [][]int, scores []Scores) []Scores {
	curLen := len(inputIDs[0])
	if curLen >= b.config.MinLength {
		return scores
	}

	eosTokenID := b.config.EOSTokenID
	for _, n := range scores {
		n.SetVec(eosTokenID, mat.Inf[mat.Float](-1))
	}

	return scores
}
