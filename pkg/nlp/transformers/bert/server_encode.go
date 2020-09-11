// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/json"
	"github.com/nlpodyssey/spago/pkg/mat"
	"net/http"
	"time"

	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
)

// SentenceEncoderHandler handles a sentence encoding request over HTTP.
func (s *Server) SentenceEncoderHandler(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*") // that's intended for testing purposes only
	w.Header().Set("Content-Type", "application/json")

	var body Body
	err := json.NewDecoder(req.Body).Decode(&body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result := s.encode(body.Text)
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

type EncodeResponse struct {
	Data []float64 `json:"data"`
	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
}

// TODO: This method is too long; it needs to be refactored.
func (s *Server) encode(text string) *EncodeResponse {
	start := time.Now()

	tokenizer := wordpiecetokenizer.New(s.model.Vocabulary)
	origTokens := tokenizer.Tokenize(text)
	tokenized := pad(tokenizers.GetStrings(origTokens))

	g := ag.NewGraph()
	defer g.Clear()
	proc := s.model.NewProc(g).(*Processor)
	proc.SetMode(nn.Inference)
	encoded := proc.Encode(tokenized)
	pooled := proc.Pool(encoded)
	normalized := pooled.Value().(*mat.Dense).Normalize2()

	return &EncodeResponse{
		Data: normalized.Data(),
		Took: time.Since(start).Milliseconds(),
	}
}
