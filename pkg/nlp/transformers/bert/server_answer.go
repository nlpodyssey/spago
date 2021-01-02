// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"context"
	"encoding/json"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"net/http"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
)

// QaHandler is the HTTP server handler function for BERT question-answering requests.
func (s *Server) QaHandler(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*") // that's intended for testing purposes only
	w.Header().Set("Content-Type", "application/json")

	var body QABody
	err := json.NewDecoder(req.Body).Decode(&body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result := s.answer(body.Question, body.Passage)
	_, pretty := req.URL.Query()["pretty"]
	response, err := Dump(result, pretty)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	_, err = w.Write(response)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
}

// Answer handles a question-answering request over gRPC.
// TODO(evanmcclure@gmail.com) Reuse the gRPC message type for HTTP requests.
func (s *Server) Answer(ctx context.Context, req *grpcapi.AnswerRequest) (*grpcapi.AnswerReply, error) {
	result := s.answer(req.GetQuestion(), req.GetPassage())

	return &grpcapi.AnswerReply{
		Answers: answersFrom(result),
		Took:    result.Took,
	}, nil
}

func answersFrom(resp *QuestionAnsweringResponse) []*grpcapi.Answer {
	result := make([]*grpcapi.Answer, len(resp.Answers))

	for i, t := range resp.Answers {
		result[i] = &grpcapi.Answer{
			Text:       t.Text,
			Start:      int32(t.Start),
			End:        int32(t.End),
			Confidence: float64(t.Confidence),
		}
	}

	return result
}

// TODO: This method is too long; it needs to be refactored.
func (s *Server) answer(question string, passage string) *QuestionAnsweringResponse {
	start := time.Now()

	tokenizer := wordpiecetokenizer.New(s.model.Vocabulary)
	origQuestionTokens := tokenizer.Tokenize(question)
	origPassageTokens := tokenizer.Tokenize(passage)

	cls := wordpiecetokenizer.DefaultClassToken
	sep := wordpiecetokenizer.DefaultSequenceSeparator
	tokenized := append([]string{cls}, append(tokenizers.GetStrings(origQuestionTokens), sep)...)
	tokenized = append(tokenized, append(tokenizers.GetStrings(origPassageTokens), sep)...)

	g := ag.NewGraph(ag.ConcurrentComputations(runtime.NumCPU()))
	defer g.Clear()
	ctx := nn.Context{Graph: g, Mode: nn.Inference}
	proc := nn.Reify(ctx, s.model).(*Model)
	encoded := proc.Encode(tokenized)

	passageStartIndex := len(origQuestionTokens) + 2 // +2 because of [CLS] and [SEP]
	passageEndIndex := passageStartIndex + len(origPassageTokens)
	startLogits, endLogits := proc.SpanClassifier.Classify(encoded)
	startLogits, endLogits = startLogits[passageStartIndex:passageEndIndex], endLogits[passageStartIndex:passageEndIndex] // cut invalid positions
	startIndices := getBestIndices(extractScores(startLogits), defaultMaxCandidateLogits)
	endIndices := getBestIndices(extractScores(endLogits), defaultMaxCandidateLogits)

	candidateAnswers := make([]Answer, 0)
	scores := make([]mat.Float, 0) // the scores are aligned with the candidateAnswers
	for _, startIndex := range startIndices {
		for _, endIndex := range endIndices {
			switch {
			case endIndex < startIndex:
				continue
			case endIndex-startIndex+1 > defaultMaxAnswerLength:
				continue
			default:
				startOffset := origPassageTokens[startIndex].Offsets.Start
				endOffset := origPassageTokens[endIndex].Offsets.End
				scores = append(scores, startLogits[startIndex].ScalarValue()+endLogits[endIndex].ScalarValue())
				candidateAnswers = append(candidateAnswers, Answer{
					Text:  strings.Trim(string([]rune(passage)[startOffset:endOffset]), " "),
					Start: startOffset,
					End:   endOffset,
				})
			}
		}
	}

	if len(candidateAnswers) == 0 {
		return &QuestionAnsweringResponse{
			Answers: AnswerSlice{},
		}
	}

	probs := floatutils.SoftMax(scores)
	answers := make(AnswerSlice, 0)
	for i, candidate := range candidateAnswers {
		if probs[i] >= defaultMinConfidence {
			candidate.Confidence = probs[i]
			answers = append(answers, candidate)
		}
	}

	sort.Sort(sort.Reverse(answers))
	if len(answers) > defaultMaxAnswers {
		answers = answers[:defaultMaxAnswers]
	}

	return &QuestionAnsweringResponse{
		Answers: answers,
		Took:    time.Since(start).Milliseconds(),
	}
}
