// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

// Package generation implements a generation search algorithm for conditional generation.
package generation

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/utils/processingqueue"
	"math"
	"sort"
	"sync"
)

// Generator is an implementation of a generation search algorithm for conditional generation.
type Generator[T mat.DType] struct {
	config          GeneratorConfig[T]
	model           EncoderDecoder[T]
	processingQueue processingqueue.ProcessingQueue
	padMask         ag.Node[T]
	eosMask         ag.Node[T]
}

// NewGenerator creates a new Generator object.
func NewGenerator[T mat.DType](config GeneratorConfig[T], model EncoderDecoder[T]) *Generator[T] {
	return &Generator[T]{
		config:          config,
		model:           model,
		processingQueue: processingqueue.New(config.MaxConcurrentComputations),
		padMask:         makePadMask(model.Graph(), config.PadTokenID, config.VocabSize),
		eosMask:         makeEosMask(model.Graph(), config.EOSTokenID, config.VocabSize),
	}
}

// Generate generates sequences for models with a language modeling head, using
// generation-search decoding.
func (b *Generator[T]) Generate(inputIDs []int) []int {
	if !b.config.IsEncoderDecoder {
		panic("generator: unsupported architecture")
	}

	encodedInput := b.model.Encode(inputIDs)
	if !b.config.IncrementalForward {
		b.performForward()
	}

	return b.beamSearch(NewScorer[T](b.config), encodedInput)
}

func (b *Generator[T]) beamSearch(scorer *Scorer[T], encodedInput []ag.Node[T]) []int {
	var (
		numBeams         = b.config.NumBeams
		beamScores       = b.makeInitBeamScores()
		decodingInputIDs = b.makeStartDecodingInputForBeamDecoding()
		scores           = make([]Scores[T], numBeams)
		cache            = make([]Cache, numBeams)
		curLen           = len(decodingInputIDs[0])
	)

	for curLen < b.config.MaxLength {
		scores, cache = b.generateNext(encodedInput, decodingInputIDs, cache)
		nextTokenScores := b.inhibitInvalidTokens(decodingInputIDs, scores)
		updateTokensScores(nextTokenScores, beamScores)
		scoredTokens := b.getTopKScoredTokens(nextTokenScores)
		beamOutputs := scorer.Process(decodingInputIDs, scoredTokens)
		beamScores = beamOutputs.nextBeamScores
		decodingInputIDs = makeNewInputIDs(decodingInputIDs, beamOutputs)
		cache = reorderCache(cache, beamOutputs.nextBeamIndices)

		if scorer.IsDone() {
			break
		}
		curLen++
	}

	return scorer.Finalize(decodingInputIDs, beamScores)
}

func (b *Generator[T]) generateNext(
	encodedInput []ag.Node[T],
	decodingInputIDs [][]int,
	pastCache []Cache,
) ([]Scores[T], []Cache) {
	numBeams := b.config.NumBeams
	logProbs := make([]ag.Node[T], numBeams)
	logits := make([]ag.Node[T], numBeams)
	scores := make([]Scores[T], numBeams)
	nextCache := make([]Cache, numBeams)

	var wg sync.WaitGroup
	wg.Add(numBeams)
	for i := 0; i < numBeams; i++ {
		i := i // redefine `i` in the inner scope, for using it in the goroutine
		b.processingQueue.Go(func() {
			defer wg.Done()
			logits[i], nextCache[i] = b.model.Decode(encodedInput, decodingInputIDs[i], pastCache[i])
			logits[i] = b.adjustLogitsDuringGeneration(logits[i], len(decodingInputIDs[i]))
			logProbs[i] = b.model.Graph().LogSoftmax(logits[i])
		})
	}
	wg.Wait()

	if !b.config.IncrementalForward {
		b.performForward()
	}

	for i := 0; i < numBeams; i++ {
		scores[i] = b.model.Graph().GetCopiedValue(logProbs[i])
	}

	return scores, nextCache
}

func (b *Generator[T]) performForward() {
	g := b.model.Graph()
	g.Forward(ag.Range[T](g.TimeStep(), -1))
	g.IncTimeStep() // mark the next block to be computed from here on
}

func (b *Generator[T]) getTopKScoredTokens(tokensScores []Scores[T]) ScoredTokens[T] {
	resultSize := b.config.NumBeams * 2
	result := make(ScoredTokens[T], 0, resultSize+1)

	var currentMinValue T = -math.MaxFloat32
	var currentMinIndex int

	for beamIndex, n := range tokensScores {
		for tokenIndex, score := range n.Data() {
			if len(result) == resultSize && score <= currentMinValue {
				continue
			}
			result = append(result, &ScoredToken[T]{
				BeamIndex:  beamIndex,
				TokenIndex: tokenIndex,
				Score:      score,
			})
			if len(result) > resultSize {
				result = append(result[:currentMinIndex], result[currentMinIndex+1:]...)
			}
			currentMinValue = math.MaxFloat32
			for ri, rv := range result {
				if rv.Score < currentMinValue {
					currentMinValue = rv.Score
					currentMinIndex = ri
				}
			}
		}
	}

	sort.SliceStable(result, func(i, j int) bool {
		return result[i].Score > result[j].Score
	})
	return result
}

func updateTokensScores[T mat.DType](tokensScores []Scores[T], beamScores []T) {
	_ = tokensScores[len(beamScores)-1]
	for i, beamScore := range beamScores {
		tokensScores[i].AddScalarInPlace(beamScore)
	}
}

func makeNewInputIDs[T mat.DType](inputIDs [][]int, scorerOut ScorerProcessOutput[T]) [][]int {
	newInputIDs := make([][]int, len(inputIDs))
	for i, beamIndex := range scorerOut.nextBeamIndices {
		prevValue := inputIDs[beamIndex]
		newInputIDs[i] = make([]int, 0, len(prevValue)+1)
		newInputIDs[i] = append(newInputIDs[i], prevValue...)
		newInputIDs[i] = append(newInputIDs[i], scorerOut.nextBeamTokens[i])
	}
	return newInputIDs
}

func reorderCache(cache []Cache, nextBeamIndices []int) []Cache {
	reorderedCache := make([]Cache, len(cache))
	for i, beamIndex := range nextBeamIndices {
		reorderedCache[i] = cache[beamIndex]
	}
	return reorderedCache
}

func (b *Generator[_]) makeStartDecodingInputForBeamDecoding() [][]int {
	beamInputIDs := make([][]int, b.config.NumBeams)
	for i := range beamInputIDs {
		beamInputIDs[i] = []int{b.config.DecoderStartTokenID}
	}
	return beamInputIDs
}

func (b *Generator[T]) makeInitBeamScores() []T {
	numBeams := b.config.NumBeams
	beamScores := make([]T, numBeams)
	beamScores[0] = 0
	for i := 1; i < numBeams; i++ {
		beamScores[i] = -1e9
	}
	return beamScores
}

func (b *Generator[T]) adjustLogitsDuringGeneration(xs ag.Node[T], curLength int) ag.Node[T] {
	// Don't generate pad token
	ys := b.model.Graph().Add(xs, b.padMask)

	if curLength == b.config.MaxLength-1 && b.config.EOSTokenID >= 0 {
		// Force EOS to be generated
		ys = b.model.Graph().Add(ys, b.eosMask)
	}
	return ys
}

func makePadMask[T mat.DType](g *ag.Graph[T], padTokenID int, vocabSize int) ag.Node[T] {
	mask := mat.NewInitVecDense[T](vocabSize, 0)
	mask.SetVec(padTokenID, mat.Inf[T](-1))
	return g.NewVariable(mask, false)
}

func makeEosMask[T mat.DType](g *ag.Graph[T], eosTokenID int, vocabSize int) ag.Node[T] {
	mask := mat.NewInitVecDense(vocabSize, mat.Inf[T](-1))
	mask.SetVec(eosTokenID, 0)
	return g.NewVariable(mask, false)
}
