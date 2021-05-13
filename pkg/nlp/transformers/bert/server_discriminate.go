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

	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
)

// DiscriminateHandler handles a discriminate request over HTTP.
func (s *Server) DiscriminateHandler(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*") // that's intended for testing purposes only
	w.Header().Set("Content-Type", "application/json")

	var body Body
	err := json.NewDecoder(req.Body).Decode(&body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result := s.discriminate(body.Text)
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

// Discriminate handles a discriminate request over gRPC.
// TODO(evanmcclure@gmail.com) Reuse the gRPC message type for HTTP requests.
func (s *Server) Discriminate(ctx context.Context, req *grpcapi.DiscriminateRequest) (*grpcapi.DiscriminateReply, error) {
	result := s.discriminate(req.GetText())

	return &grpcapi.DiscriminateReply{
		Tokens: tokensFrom(result.Tokens),
		Took:   result.Took,
	}, nil
}

func tokensFrom(tokens []Token) []*grpcapi.Token {
	result := make([]*grpcapi.Token, len(tokens))
	for i, t := range tokens {
		result[i] = &grpcapi.Token{
			Text:  t.Text,
			Start: int32(t.Start),
			End:   int32(t.End),
			Label: t.Label,
		}
	}
	return result
}

// TODO: This method is too long; it needs to be refactored.
func (s *Server) discriminate(text string) *Response {
	start := time.Now()

	tokenizer := wordpiecetokenizer.New(s.model.Vocabulary)
	origTokens := tokenizer.Tokenize(text)
	groupedTokens := wordpiecetokenizer.GroupPieces(origTokens)
	tokenized := pad(tokenizers.GetStrings(origTokens))

	g := ag.NewGraph(ag.ConcurrentComputations(runtime.NumCPU()))
	defer g.Clear()
	proc := nn.ReifyForInference(s.model, g).(*Model)
	encoded := proc.Encode(tokenized)

	fakeTokens := make(map[int]bool, 0)
	for i, fake := range proc.Discriminate(encoded) {
		if i == 0 || i == len(tokenized)-1 {
			continue // skip padding
		}
		if fake == 1.0 {
			fakeTokens[i-1] = true // -1 because of [CLS]
		}
	}

	fakeCompleteWords := make([]bool, len(groupedTokens))
	for i, group := range groupedTokens {
		fakeCompleteWords[i] = false
		for j := group.Start; j <= group.End; j++ {
			if fakeTokens[j] {
				fakeCompleteWords[i] = true
			}
		}
	}

	retTokens := make([]Token, 0)
	for i := range groupedTokens {
		label := DefaultRealLabel
		if fakeCompleteWords[i] {
			label = DefaultFakeLabel
		}
		group := groupedTokens[i]
		startToken, endToken := origTokens[group.Start], origTokens[group.End]
		retTokens = append(retTokens, Token{
			Text:  string([]rune(text)[startToken.Offsets.Start:endToken.Offsets.End]),
			Start: startToken.Offsets.Start,
			End:   endToken.Offsets.End,
			Label: label,
		})
	}
	return &Response{Tokens: retTokens, Took: time.Since(start).Milliseconds()}
}
