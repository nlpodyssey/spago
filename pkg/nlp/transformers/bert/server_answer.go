// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"context"
	"encoding/json"
	"github.com/nlpodyssey/spago/pkg/mat"
	"net/http"
	"time"

	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
)

// QaHandler is the HTTP server handler function for BERT question-answering requests.
func (s *Server[T]) QaHandler(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*") // that's intended for testing purposes only
	w.Header().Set("Content-Type", "application/json")

	var body QABody
	err := json.NewDecoder(req.Body).Decode(&body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	start := time.Now()
	answers := s.model.Answer(body.Question, body.Passage)
	result := &QuestionAnsweringResponse[T]{
		Answers: answers,
		Took:    time.Since(start).Milliseconds(),
	}

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
func (s *Server[T]) Answer(_ context.Context, req *grpcapi.AnswerRequest) (*grpcapi.AnswerReply, error) {
	start := time.Now()
	result := s.model.Answer(req.GetQuestion(), req.GetPassage())
	return &grpcapi.AnswerReply{
		Answers: answersFrom(result),
		Took:    time.Since(start).Milliseconds(),
	}, nil
}

func answersFrom[T mat.DType](answers Answers[T]) []*grpcapi.Answer {
	result := make([]*grpcapi.Answer, len(answers))
	for i, a := range answers {
		result[i] = &grpcapi.Answer{
			Text:       a.Text,
			Start:      int32(a.Start),
			End:        int32(a.End),
			Confidence: float64(a.Confidence),
		}
	}
	return result
}
