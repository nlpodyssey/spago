// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sequencelabeler

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/basetokenizer"
	"github.com/nlpodyssey/spago/pkg/utils/httphandlers"
)

type Server struct {
	model *Model
	port  int
}

func NewServer(model *Model, port int) *Server {
	return &Server{
		model: model,
		port:  port,
	}
}

func (s *Server) Start() {
	r := http.NewServeMux()
	r.HandleFunc("/analyze", s.analyze)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", s.port),
		httphandlers.RecoveryHandler(httphandlers.PrintRecoveryStack(true))(r)))
}

type OptionsType struct {
	MergeEntities     bool `json:"mergeEntities"`     // default false
	FilterNotEntities bool `json:"filterNotEntities"` // default false
}

type Body struct {
	Options OptionsType `json:"options"`
	Text    string      `json:"text"`
}

func (s *Server) analyze(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*") // that's intended for testing purposes only
	w.Header().Set("Content-Type", "application/json")

	var body Body
	err := json.NewDecoder(req.Body).Decode(&body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	analysis, took := s.process(body.Text, body.Options.MergeEntities)
	if body.Options.FilterNotEntities {
		analysis = filterNotEntities(analysis)
	}
	result := prepareResponse(analysis, took)

	_, pretty := req.URL.Query()["pretty"]
	response, err := result.Dump(pretty)
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

func (s *Server) process(text string, merge bool) ([]TokenLabel, time.Duration) {
	start := time.Now()
	g := ag.NewGraph()
	defer g.Clear()
	proc := s.model.NewProc(g).(*Processor)
	proc.SetMode(nn.Inference)
	tokenized := basetokenizer.New().Tokenize(text)
	predicted := proc.Predict(tokenized)
	if merge {
		predicted = mergeEntities(predicted)
	}
	return predicted, time.Since(start)
}

func prepareResponse(tokens []TokenLabel, took time.Duration) *Response {
	newTokens := make([]Token, len(tokens))
	for i, token := range tokens {
		newTokens[i] = Token{
			Text:  token.String,
			Start: token.Offsets.Start,
			End:   token.Offsets.End,
			Label: token.Label,
		}
	}
	return &Response{Tokens: newTokens, Took: took.Milliseconds()}
}

type Response struct {
	Tokens []Token `json:"tokens"`
	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
}

type Token struct {
	Text  string `json:"text"`
	Start int    `json:"start"`
	End   int    `json:"end"`
	Label string `json:"label"`
}

func (r *Response) Dump(pretty bool) ([]byte, error) {
	buf := bytes.NewBufferString("")
	enc := json.NewEncoder(buf)
	if pretty {
		enc.SetIndent("", "    ")
	}
	enc.SetEscapeHTML(true)
	err := enc.Encode(r)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
