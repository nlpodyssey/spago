// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"github.com/nlpodyssey/spago/pkg/mat/matutils"
	matsort "github.com/nlpodyssey/spago/pkg/mat/sort"
	"runtime"
	"sort"
	"strings"

	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
)

const defaultMaxAnswerLength = 20     // TODO: from options
const defaultMinConfidence = 0.1      // TODO: from options
const defaultMaxCandidateLogits = 3.0 // TODO: from options
const defaultMaxAnswers = 3           // TODO: from options

// Answer represent a single JSON-serializable BERT question-answering answer,
// used as part of a server's response.
type Answer[T mat.DType] struct {
	Text       string `json:"text"`
	Start      int    `json:"start"`
	End        int    `json:"end"`
	Confidence T      `json:"confidence"`
}

// Answers is a slice of Answer elements, which implements the sort.Interface.
type Answers[T mat.DType] []Answer[T]

// Len returns the length of the slice.
func (a Answers[_]) Len() int {
	return len(a)
}

// Less returns true if the Answer.Confidence of the element at position i is
// lower than the one of the element at position j.
func (a Answers[_]) Less(i, j int) bool {
	return a[i].Confidence < a[j].Confidence
}

// Swap swaps the elements at positions i and j.
func (a Answers[_]) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

// Sort sorts the Answers's elements by Answer.Confidence.
func (a Answers[_]) Sort() {
	sort.Sort(a)
}

// Answer returns a slice of candidate answers for the given question-passage pair.
// The answers are sorted by confidence level in descending order.
func (m *Model[T]) Answer(question string, passage string) Answers[T] {
	tokenizer := wordpiecetokenizer.New(m.Vocabulary)
	questionTokens := tokenizer.Tokenize(question)
	passageTokens := tokenizer.Tokenize(passage)
	tokenized := mixQuestionAndPassageTokens(questionTokens, passageTokens)

	g := ag.NewGraph[T](ag.ConcurrentComputations[T](runtime.NumCPU()))
	defer g.Clear()
	proc := nn.ReifyForInference(m, g)
	encoded := proc.Encode(tokenized)

	startLogits, endLogits := proc.SpanClassifier.Classify(encoded)
	startLogits, endLogits = adjustLogitsForInference(startLogits, endLogits, questionTokens, passageTokens)

	startIndices := getBestIndices(extractScores(startLogits), defaultMaxCandidateLogits)
	endIndices := getBestIndices(extractScores(endLogits), defaultMaxCandidateLogits)

	candidateAnswers, scores := searchCandidateAnswers(
		startIndices, endIndices,
		startLogits, endLogits,
		passageTokens,
		passage,
	)

	if len(candidateAnswers) == 0 {
		return nil
	}

	answers := assignScoresAndFilterUnlikelyCandidates(candidateAnswers, scores)

	sort.Sort(sort.Reverse(answers))
	if len(answers) > defaultMaxAnswers {
		answers = answers[:defaultMaxAnswers]
	}
	return answers
}

func mixQuestionAndPassageTokens(question, passage []tokenizers.StringOffsetsPair) []string {
	cls := wordpiecetokenizer.DefaultClassToken
	sep := wordpiecetokenizer.DefaultSequenceSeparator
	tokenized := append([]string{cls}, append(tokenizers.GetStrings(question), sep)...)
	tokenized = append(tokenized, append(tokenizers.GetStrings(passage), sep)...)
	return tokenized
}

func adjustLogitsForInference[T mat.DType](
	startLogits, endLogits []ag.Node[T],
	question, passage []tokenizers.StringOffsetsPair,
) ([]ag.Node[T], []ag.Node[T]) {
	passageStartIndex := len(question) + 2 // +2 because of [CLS] and [SEP]
	passageEndIndex := passageStartIndex + len(passage)
	return startLogits[passageStartIndex:passageEndIndex], endLogits[passageStartIndex:passageEndIndex] // cut invalid positions
}

func getBestIndices[T mat.DType](logits []T, size int) []int {
	s := matsort.NewDTSlice(logits...)
	sort.Sort(sort.Reverse(s))
	if len(s.Indices) < size {
		return s.Indices
	}
	return s.Indices[:size]
}

func extractScores[T mat.DType](logits []ag.Node[T]) []T {
	scores := make([]T, len(logits))
	for i, node := range logits {
		scores[i] = node.ScalarValue()
	}
	return scores
}

func searchCandidateAnswers[T mat.DType](
	startIndices, endIndices []int,
	startLogits, endLogits []ag.Node[T],
	passageTokens []tokenizers.StringOffsetsPair,
	passage string,
) (Answers[T], []T) {
	candidateAnswers := make(Answers[T], 0)
	scores := make([]T, 0) // the scores are aligned with the candidateAnswers
	for _, startIndex := range startIndices {
		for _, endIndex := range endIndices {
			switch {
			case endIndex < startIndex:
				continue
			case endIndex-startIndex+1 > defaultMaxAnswerLength:
				continue
			default:
				startOffset := passageTokens[startIndex].Offsets.Start
				endOffset := passageTokens[endIndex].Offsets.End
				scores = append(scores, startLogits[startIndex].ScalarValue()+endLogits[endIndex].ScalarValue())
				candidateAnswers = append(candidateAnswers, Answer[T]{
					Text:  strings.Trim(string([]rune(passage)[startOffset:endOffset]), " "),
					Start: startOffset,
					End:   endOffset,
				})
			}
		}
	}

	return candidateAnswers, scores
}

func assignScoresAndFilterUnlikelyCandidates[T mat.DType](candidates Answers[T], scores []T) Answers[T] {
	probs := matutils.SoftMax(scores)
	answers := make(Answers[T], 0)
	for i, candidate := range candidates {
		if probs[i] >= defaultMinConfidence {
			candidate.Confidence = probs[i]
			answers = append(answers, candidate)
		}
	}
	return answers
}
