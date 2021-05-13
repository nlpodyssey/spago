// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	matsort "github.com/nlpodyssey/spago/pkg/mat32/sort"
	"runtime"
	"sort"
	"strings"

	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
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
type Answer struct {
	Text       string    `json:"text"`
	Start      int       `json:"start"`
	End        int       `json:"end"`
	Confidence mat.Float `json:"confidence"`
}

// Answers is a slice of Answer elements, which implements the sort.Interface.
type Answers []Answer

// Len returns the length of the slice.
func (p Answers) Len() int {
	return len(p)
}

// Less returns true if the Answer.Confidence of the element at position i is
// lower than the one of the element at position j.
func (p Answers) Less(i, j int) bool {
	return p[i].Confidence < p[j].Confidence
}

// Swap swaps the elements at positions i and j.
func (p Answers) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

// Sort sorts the Answers's elements by Answer.Confidence.
func (p Answers) Sort() {
	sort.Sort(p)
}

// Answer returns a slice of candidate answers for the given question-passage pair.
// The answers are sorted by confidence level in descending order.
func (m *Model) Answer(question string, passage string) Answers {
	tokenizer := wordpiecetokenizer.New(m.Vocabulary)
	questionTokens := tokenizer.Tokenize(question)
	passageTokens := tokenizer.Tokenize(passage)
	tokenized := mixQuestionAndPassageTokens(questionTokens, passageTokens)

	g := ag.NewGraph(ag.ConcurrentComputations(runtime.NumCPU()))
	defer g.Clear()
	proc := nn.ReifyForInference(m, g).(*Model)
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

func adjustLogitsForInference(
	startLogits, endLogits []ag.Node,
	question, passage []tokenizers.StringOffsetsPair,
) ([]ag.Node, []ag.Node) {
	passageStartIndex := len(question) + 2 // +2 because of [CLS] and [SEP]
	passageEndIndex := passageStartIndex + len(passage)
	return startLogits[passageStartIndex:passageEndIndex], endLogits[passageStartIndex:passageEndIndex] // cut invalid positions
}

func getBestIndices(logits []mat.Float, size int) []int {
	s := matsort.NewFloatSlice(logits...)
	sort.Sort(sort.Reverse(s))
	if len(s.Indices) < size {
		return s.Indices
	}
	return s.Indices[:size]
}

func extractScores(logits []ag.Node) []mat.Float {
	scores := make([]mat.Float, len(logits))
	for i, node := range logits {
		scores[i] = node.ScalarValue()
	}
	return scores
}

func searchCandidateAnswers(
	startIndices, endIndices []int,
	startLogits, endLogits []ag.Node,
	passageTokens []tokenizers.StringOffsetsPair,
	passage string,
) (Answers, []mat.Float) {
	candidateAnswers := make(Answers, 0)
	scores := make([]mat.Float, 0) // the scores are aligned with the candidateAnswers
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
				candidateAnswers = append(candidateAnswers, Answer{
					Text:  strings.Trim(string([]rune(passage)[startOffset:endOffset]), " "),
					Start: startOffset,
					End:   endOffset,
				})
			}
		}
	}

	return candidateAnswers, scores
}

func assignScoresAndFilterUnlikelyCandidates(candidates Answers, scores []mat.Float) Answers {
	probs := floatutils.SoftMax(scores)
	answers := make(Answers, 0)
	for i, candidate := range candidates {
		if probs[i] >= defaultMinConfidence {
			candidate.Confidence = probs[i]
			answers = append(answers, candidate)
		}
	}
	return answers
}
