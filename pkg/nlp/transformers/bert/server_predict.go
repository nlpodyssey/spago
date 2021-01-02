// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"context"
	"encoding/json"
	"net/http"
	"runtime"
	"time"

	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
)

// PredictHandler handles a predict request over HTTP.
func (s *Server) PredictHandler(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*") // that's intended for testing purposes only
	w.Header().Set("Content-Type", "application/json")

	var body Body
	err := json.NewDecoder(req.Body).Decode(&body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result := s.predict(body.Text)
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

// Predict handles a predict request over gRPC.
// TODO(evanmcclure@gmail.com) Reuse the gRPC message type for HTTP requests.
func (s *Server) Predict(ctx context.Context, req *grpcapi.PredictRequest) (*grpcapi.PredictReply, error) {
	result := s.predict(req.GetText())

	return &grpcapi.PredictReply{
		Tokens: tokensFrom(result),
		Took:   result.Took,
	}, nil
}

// TODO: This method is too long; it needs to be refactored.
func (s *Server) predict(text string) *Response {
	start := time.Now()

	tokenizer := wordpiecetokenizer.New(s.model.Vocabulary)
	origTokens := tokenizer.Tokenize(text)
	tokenized := pad(tokenizers.GetStrings(origTokens))

	g := ag.NewGraph(ag.ConcurrentComputations(runtime.NumCPU()))
	defer g.Clear()
	proc := nn.Reify(nn.Context{Graph: g, Mode: nn.Inference}, s.model).(*Model)
	encoded := proc.Encode(tokenized)

	masked := make([]int, 0)
	for i := range tokenized {
		if tokenized[i] == wordpiecetokenizer.DefaultMaskToken {
			masked = append(masked, i)
		}
	}

	retTokens := make([]Token, 0)
	for tokenID, prediction := range proc.PredictMasked(encoded, masked) {
		bestPredictedWordIndex := floatutils.ArgMax(prediction.Value().Data())
		word, ok := s.model.Vocabulary.Term(bestPredictedWordIndex)
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
	return &Response{Tokens: retTokens, Took: time.Since(start).Milliseconds()}
}
