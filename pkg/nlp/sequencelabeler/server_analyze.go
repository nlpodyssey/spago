// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sequencelabeler

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/nlpodyssey/spago/pkg/nlp/sequencelabeler/grpcapi"
)

// OptionsType provides JSON-serializable options for the sequence labeling Server.
type OptionsType struct {
	MergeEntities     bool `json:"mergeEntities"`     // default false
	FilterNotEntities bool `json:"filterNotEntities"` // default false
}

// Body provides JSON-serializable parameters for sequence labeling Server requests.
type Body struct {
	Options OptionsType `json:"options"`
	Text    string      `json:"text"`
}

// Response provides JSON-serializable parameters for sequence labeling Server responses.
type Response struct {
	Tokens []Token `json:"tokens"`
	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
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

	start := time.Now()
	analysis := s.model.Analyze(body.Text, body.Options.MergeEntities, body.Options.FilterNotEntities)

	result := &Response{
		Tokens: analysis.Tokens,
		Took:   time.Since(start).Milliseconds(),
	}

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

// Analyze sends a request to /analyze.
// TODO(evanmcclure@gmail.com) Reuse the gRPC message type for HTTP requests.
func (s *Server) Analyze(_ context.Context, req *grpcapi.AnalyzeRequest) (*grpcapi.AnalyzeReply, error) {
	start := time.Now()
	analysis := s.model.Analyze(
		req.GetText(),
		req.GetMergeEntities(),
		req.GetFilterNotEntities(),
	)
	return &grpcapi.AnalyzeReply{
		Tokens: tokensFrom(analysis.Tokens),
		Took:   time.Since(start).Milliseconds(),
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

// Dump serializes the Response to JSON.
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
