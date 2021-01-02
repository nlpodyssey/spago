// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"bytes"
	"encoding/json"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	matsort "github.com/nlpodyssey/spago/pkg/mat32/sort"
	"github.com/nlpodyssey/spago/pkg/webui/bertclassification"
	"net/http"
	"sort"

	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
	"github.com/nlpodyssey/spago/pkg/utils/grpcutils"
	"github.com/nlpodyssey/spago/pkg/utils/httputils"
	"github.com/nlpodyssey/spago/pkg/webui/bertqa"
)

// TODO: This code needs to be refactored. Pull requests are welcome!

// Server contains everything needed to run a BERT server.
type Server struct {
	model *Model

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

	go httputils.RunHTTPServer(address, tlsDisable, tlsCert, tlsKey, mux)

	grpcServer := grpcutils.NewGRPCServer(tlsDisable, tlsCert, tlsKey)
	grpcapi.RegisterBERTServer(grpcServer, s)
	grpcutils.RunGRPCServer(grpcAddress, grpcServer)
}

// Body is the JSON-serializable expected request body for various BERT server requests.
type Body struct {
	Text  string `json:"text"`
	Text2 string `json:"text2"`
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

// Answer represent a single JSON-serializable BERT question-answering answer,
// used as part of a server's response.
type Answer struct {
	Text       string    `json:"text"`
	Start      int       `json:"start"`
	End        int       `json:"end"`
	Confidence mat.Float `json:"confidence"`
}

// AnswerSlice is a slice of Answer elements, which implements the sort.Interface.
type AnswerSlice []Answer

// Len returns the length of the slice.
func (p AnswerSlice) Len() int {
	return len(p)
}

// Less returns true if the Answer.Confidence of the element at position i is
// lower than the one of the element at position j.
func (p AnswerSlice) Less(i, j int) bool {
	return p[i].Confidence < p[j].Confidence
}

// Swap swaps the elements at positions i and j.
func (p AnswerSlice) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

// Sort sorts the AnswerSlice's elements by Answer.Confidence.
func (p AnswerSlice) Sort() {
	sort.Sort(p)
}

// QuestionAnsweringResponse is the JSON-serializable structure for BERT
// question-answering server response.
type QuestionAnsweringResponse struct {
	Answers AnswerSlice `json:"answers"`
	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
}

const defaultMaxAnswerLength = 20     // TODO: from options
const defaultMinConfidence = 0.1      // TODO: from options
const defaultMaxCandidateLogits = 3.0 // TODO: from options
const defaultMaxAnswers = 3           // TODO: from options

func extractScores(logits []ag.Node) []mat.Float {
	scores := make([]mat.Float, len(logits))
	for i, node := range logits {
		scores[i] = node.ScalarValue()
	}
	return scores
}

func getBestIndices(logits []mat.Float, size int) []int {
	s := matsort.NewFloatSlice(logits...)
	sort.Sort(sort.Reverse(s))
	if len(s.Indices) < size {
		return s.Indices
	}
	return s.Indices[:size]
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
