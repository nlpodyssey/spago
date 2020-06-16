// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/nlpodyssey/spago/pkg/utils/httphandlers"
)

// TODO: This code needs to be refactored. Pull requests are welcome!

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
	r.HandleFunc("/discriminate", s.discriminateHandler)
	r.HandleFunc("/predict", s.predictHandler)
	r.HandleFunc("/answer", s.qaHandler)
	// r.HandleFunc("/classify", s.classifyHandler)
	// r.HandleFunc("/tag", s.tagHandler)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", s.port),
		httphandlers.RecoveryHandler(httphandlers.PrintRecoveryStack(true))(r)))
}

type Body struct {
	Text string `json:"text"`
}

func (s *Server) discriminateHandler(w http.ResponseWriter, req *http.Request) {
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

func (s *Server) predictHandler(w http.ResponseWriter, req *http.Request) {
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

type QABody struct {
	Question string `json:"question"`
	Passage  string `json:"passage"`
}

func (s *Server) qaHandler(w http.ResponseWriter, req *http.Request) {
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

func pad(words []string) []string {
	leftPad := wordpiecetokenizer.DefaultClassToken
	rightPad := wordpiecetokenizer.DefaultSequenceSeparator
	return append([]string{leftPad}, append(words, rightPad)...)
}

const DefaultRealLabel = "REAL"
const DefaultFakeLabel = "FAKE"
const DefaultPredictedLabel = "PREDICTED"

// TODO: This method is too long; it needs to be refactored.
func (s *Server) discriminate(text string) *Response {
	start := time.Now()

	tokenizer := wordpiecetokenizer.New(s.model.Vocabulary)
	origTokens := tokenizer.Tokenize(text)
	groupedTokens := wordpiecetokenizer.GroupPieces(origTokens)
	tokenized := pad(tokenizers.GetStrings(origTokens))

	g := ag.NewGraph()
	defer g.Clear()
	proc := s.model.NewProc(g).(*Processor)
	proc.SetMode(nn.Inference)
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

// TODO: This method is too long; it needs to be refactored.
func (s *Server) predict(text string) *Response {
	start := time.Now()

	tokenizer := wordpiecetokenizer.New(s.model.Vocabulary)
	origTokens := tokenizer.Tokenize(text)
	tokenized := pad(tokenizers.GetStrings(origTokens))

	g := ag.NewGraph()
	defer g.Clear()
	proc := s.model.NewProc(g).(*Processor)
	proc.SetMode(nn.Inference)
	encoded := proc.Encode(tokenized)

	masked := make([]int, 0)
	for i := range tokenized {
		if tokenized[i] == wordpiecetokenizer.DefaultMaskToken {
			masked = append(masked, i)
		}
	}

	retTokens := make([]Token, 0)
	for tokenId, prediction := range proc.PredictMasked(encoded, masked) {
		bestPredictedWordIndex := f64utils.ArgMax(prediction.Value().Data())
		word, ok := s.model.Vocabulary.Term(bestPredictedWordIndex)
		if !ok {
			word = wordpiecetokenizer.DefaultUnknownToken // if this is returned, there's a misalignment with the vocabulary
		}
		label := DefaultPredictedLabel
		retTokens = append(retTokens, Token{
			Text:  word,
			Start: origTokens[tokenId-1].Offsets.Start, // skip CLS
			End:   origTokens[tokenId-1].Offsets.End,   // skip CLS
			Label: label,
		})
	}
	return &Response{Tokens: retTokens, Took: time.Since(start).Milliseconds()}
}

type Answer struct {
	Text       string  `json:"text"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
	Confidence float64 `json:"confidence"`
}

type AnswerSlice []Answer

func (p AnswerSlice) Len() int           { return len(p) }
func (p AnswerSlice) Less(i, j int) bool { return p[i].Confidence < p[j].Confidence }
func (p AnswerSlice) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p AnswerSlice) Sort()              { sort.Sort(p) }

type QuestionAnsweringResponse struct {
	Answers AnswerSlice `json:"answers"`
	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
}

func (r *QuestionAnsweringResponse) Dump(pretty bool) ([]byte, error) {
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

const defaultMaxAnswerLength = 20     // TODO: from options
const defaultMinConfidence = 0.1      // TODO: from options
const defaultMaxCandidateLogits = 3.0 // TODO: from options
const defaultMaxAnswers = 3           // TODO: from options

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

	g := ag.NewGraph()
	defer g.Clear()
	proc := s.model.NewProc(g).(*Processor)
	proc.SetMode(nn.Inference)
	encoded := proc.Encode(tokenized)

	passageStartIndex := len(origQuestionTokens) + 2
	startLogits, endLogits := proc.SpanClassifier.Classify(encoded)
	startLogits, endLogits = startLogits[passageStartIndex:], endLogits[passageStartIndex:] // cut invalid positions
	startIndices := getBestIndices(extractScores(startLogits), defaultMaxCandidateLogits)
	endIndices := getBestIndices(extractScores(endLogits), defaultMaxCandidateLogits)

	candidateAnswers := make([]Answer, 0)
	scores := make([]float64, 0) // the scores are aligned with the candidateAnswers
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

	probs := f64utils.SoftMax(scores)
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

func extractScores(logits []ag.Node) []float64 {
	scores := make([]float64, len(logits))
	for i, node := range logits {
		scores[i] = node.ScalarValue()
	}
	return scores
}

func getBestIndices(logits []float64, size int) []int {
	s := utils.NewFloat64Slice(logits...)
	sort.Sort(sort.Reverse(s))
	if len(s.Indices) < size {
		return s.Indices
	}
	return s.Indices[:size]
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
