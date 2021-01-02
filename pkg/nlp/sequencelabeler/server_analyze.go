// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sequencelabeler

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"runtime"
	"time"

	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/sequencelabeler/grpcapi"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/basetokenizer"
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

// Analyze sends a request to /analyze.
// TODO(evanmcclure@gmail.com) Reuse the gRPC message type for HTTP requests.
func (s *Server) Analyze(ctx context.Context, req *grpcapi.AnalyzeRequest) (*grpcapi.AnalyzeReply, error) {
	analysis, took := s.process(req.GetText(), req.GetMergeEntities())
	if req.GetFilterNotEntities() {
		analysis = filterNotEntities(analysis)
	}
	result := prepareResponse(analysis, took)

	return &grpcapi.AnalyzeReply{
		Tokens: tokensFrom(result),
		Took:   took.Milliseconds(),
	}, nil
}

func tokensFrom(resp *Response) []*grpcapi.Token {
	result := make([]*grpcapi.Token, len(resp.Tokens))

	for i, t := range resp.Tokens {
		result[i] = &grpcapi.Token{
			Text:  t.Text,
			Start: int32(t.Start),
			End:   int32(t.End),
			Label: t.Label,
		}
	}

	return result
}

func (s *Server) process(text string, merge bool) ([]TokenLabel, time.Duration) {
	start := time.Now()
	g := ag.NewGraph(ag.ConcurrentComputations(runtime.NumCPU()))
	defer g.Clear()
	proc := nn.Reify(nn.Context{Graph: g, Mode: nn.Inference}, s.model).(*Model)
	tokenized := basetokenizer.New().Tokenize(text)
	predicted := proc.Forward(tokenized)
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

// Response provides JSON-serializable parameters for sequence labeling Server responses.
type Response struct {
	Tokens []Token `json:"tokens"`
	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
}

// Token provides JSON-serializable parameters for a single token of sequence
// labeling Server responses.
type Token struct {
	Text  string `json:"text"`
	Start int    `json:"start"`
	End   int    `json:"end"`
	Label string `json:"label"`
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
