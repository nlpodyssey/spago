// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"bytes"
	"encoding/json"
	"github.com/nlpodyssey/spago/pkg/webui/bertclassification"
	"net/http"
	"sort"

	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
	"github.com/nlpodyssey/spago/pkg/utils/grpcutils"
	"github.com/nlpodyssey/spago/pkg/utils/httputils"
	"github.com/nlpodyssey/spago/pkg/webui/bertqa"
)

// TODO: This code needs to be refactored. Pull requests are welcome!

// Server contains everything needed to run a BERT server.
type Server struct {
	model           *Model
	TimeoutSeconds  int
	MaxRequestBytes int

	// UnimplementedBERTServer must be embedded to have forward compatible implementations for gRPC.
	grpcapi.UnimplementedBERTServer
}

// NewServer returns Server objects.
func NewServer(model *Model) *Server {
	return &Server{
		model: model,
	}
}

// StartDefaultServer is used to start a basic BERT HTTP server.
// If you want more control of the HTTP server you can run your own
// HTTP router using the public handler functions
func (s *Server) StartDefaultServer(address, grpcAddress, tlsCert, tlsKey string, tlsDisable bool) {
	mux := http.NewServeMux()
	mux.HandleFunc("/bert-qa-ui", bertqa.Handler)
	mux.HandleFunc("/bert-classify-ui", bertclassification.Handler)
	mux.HandleFunc("/discriminate", s.DiscriminateHandler)
	mux.HandleFunc("/predict", s.PredictHandler)
	mux.HandleFunc("/answer", s.QaHandler)
	mux.HandleFunc("/tag", s.LabelerHandler)
	mux.HandleFunc("/classify", s.ClassifyHandler)
	mux.HandleFunc("/encode", s.SentenceEncoderHandler)

	go httputils.RunHTTPServer(httputils.HTTPServerConfig{
		Address:         address,
		TLSDisable:      tlsDisable,
		TLSCert:         tlsCert,
		TLSKey:          tlsKey,
		TimeoutSeconds:  s.TimeoutSeconds,
		MaxRequestBytes: s.MaxRequestBytes,
	}, mux)

	grpcServer := grpcutils.NewGRPCServer(grpcutils.GRPCServerConfig{
		TLSDisable:      tlsDisable,
		TLSCert:         tlsCert,
		TLSKey:          tlsKey,
		TimeoutSeconds:  s.TimeoutSeconds,
		MaxRequestBytes: s.MaxRequestBytes,
	})
	grpcapi.RegisterBERTServer(grpcServer, s)
	grpcutils.RunGRPCServer(grpcAddress, grpcServer)
}

// Body is the JSON-serializable expected request body for various BERT server requests.
type Body struct {
	Text            string                                `json:"text"`
	Text2           string                                `json:"text2"`
	PoolingStrategy grpcapi.EncodeRequest_PoolingStrategy `json:"pooling_strategy"`
}

// QABody is the JSON-serializable expected request body for BERT question-answering server requests.
type QABody struct {
	Question string `json:"question"`
	Passage  string `json:"passage"`
}

func pad(words []string) []string {
	leftPad := wordpiecetokenizer.DefaultClassToken
	rightPad := wordpiecetokenizer.DefaultSequenceSeparator
	return append([]string{leftPad}, append(words, rightPad)...)
}

// DefaultRealLabel is the default value for the real label used for BERT
// "discriminate" server requests.
const DefaultRealLabel = "REAL"

// DefaultFakeLabel is the default value for the fake label used for BERT
// "discriminate" server requests.
const DefaultFakeLabel = "FAKE"

// DefaultPredictedLabel is the default value for the predicted label used for BERT
// "predict" server requests.
const DefaultPredictedLabel = "PREDICTED"

// QuestionAnsweringResponse is the JSON-serializable structure for BERT
// question-answering server response.
type QuestionAnsweringResponse struct {
	Answers Answers `json:"answers"`
	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
}

// Response is the JSON-serializable server response for various BERT-related requests.
type Response struct {
	Tokens []Token `json:"tokens"`
	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
}

// Token is a JSON-serializable labeled text token.
type Token struct {
	Text  string `json:"text"`
	Start int    `json:"start"`
	End   int    `json:"end"`
	Label string `json:"label"`
}

// TokenSlice is a slice of Token elements, which implements the sort.Interface.
type TokenSlice []Token

// Len returns the length of the slice.
func (p TokenSlice) Len() int {
	return len(p)
}

// Less returns true if the Token.Start of the element at position i is
// lower than the one of the element at position j.
func (p TokenSlice) Less(i, j int) bool {
	return p[i].Start < p[j].Start
}

// Swap swaps the elements at positions i and j.
func (p TokenSlice) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

// Sort sorts the TokenSlice's elements by Token.Start.
func (p TokenSlice) Sort() {
	sort.Sort(p)
}

// Dump serializes the given value to JSON.
func Dump(value interface{}, pretty bool) ([]byte, error) {
	buf := bytes.NewBufferString("")
	enc := json.NewEncoder(buf)
	if pretty {
		enc.SetIndent("", "    ")
	}
	enc.SetEscapeHTML(true)
	err := enc.Encode(value)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
