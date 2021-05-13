// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/json"
	"fmt"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"net/http"
	"strings"
	"time"

	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
)

// LabelerOptionsType is a JSON-serializable set of options for BERT "tag" (labeler) requests.
type LabelerOptionsType struct {
	MergeEntities     bool `json:"mergeEntities"`     // default false
	FilterNotEntities bool `json:"filterNotEntities"` // default false
}

// TokenClassifierBody provides JSON-serializable parameters for BERT "tag" (labeler) requests.
type TokenClassifierBody struct {
	Options LabelerOptionsType `json:"options"`
	Text    string             `json:"text"`
}

// LabelerHandler handles a labeling request over HTTP.
func (s *Server) LabelerHandler(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*") // that's intended for testing purposes only
	w.Header().Set("Content-Type", "application/json")

	var body TokenClassifierBody
	err := json.NewDecoder(req.Body).Decode(&body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result := s.label(body.Text, body.Options.MergeEntities, body.Options.FilterNotEntities)

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

// TODO: This method is too long; it needs to be refactored.
func (s *Server) label(text string, merge bool, filter bool) *Response {
	start := time.Now()

	tokenizer := wordpiecetokenizer.New(s.model.Vocabulary)
	origTokens := tokenizer.Tokenize(text)
	tokensRange := wordpiecetokenizer.GroupPieces(origTokens)
	groupedTokens := wordpiecetokenizer.MakeOffsetPairsFromGroups(text, origTokens, tokensRange)
	tokenized := pad(tokenizers.GetStrings(origTokens))

	g := ag.NewGraph()
	defer g.Clear()
	proc := nn.ReifyForInference(s.model, g).(*Model)
	encoded := proc.Encode(tokenized)
	encoded = encoded[1 : len(encoded)-1] // trim [CLS] and [SEP]

	// average pooling
	avgEncoded := make([]ag.Node, len(tokensRange))
	for i, group := range tokensRange {
		cnt := 0
		for j := group.Start; j <= group.End; j++ {
			avgEncoded[i] = g.Add(avgEncoded[i], encoded[j])
			cnt++
		}
		if cnt > 1 {
			avgEncoded[i] = g.DivScalar(avgEncoded[i], g.NewScalar(mat.Float(cnt)))
		}
	}

	retTokens := make([]Token, 0)
	for i, logits := range proc.TokenClassification(avgEncoded) {
		probs := floatutils.SoftMax(logits.Value().Data())
		best := floatutils.ArgMax(probs)
		retTokens = append(retTokens, Token{
			Text:  groupedTokens[i].String,
			Start: groupedTokens[i].Offsets.Start,
			End:   groupedTokens[i].Offsets.End,
			Label: s.model.Classifier.Config.Labels[best],
		})
	}
	if merge {
		retTokens = mergeEntities(text, retTokens)
	}
	if filter {
		retTokens = filterNotEntities(retTokens)
	}
	return &Response{Tokens: retTokens, Took: time.Since(start).Milliseconds()}
}

// TODO: make sure that the input label sequence is valid
func mergeEntities(text string, tokens []Token) []Token {
	newTokens := make([]Token, 0)
	var buf *tokenizers.StringOffsetsPair
	flush := func() {
		if buf != nil {
			startOffset := buf.Offsets.Start
			endOffset := buf.Offsets.End
			newTokens = append(newTokens, Token{
				Text:  strings.Trim(string([]rune(text)[startOffset:endOffset]), " "),
				Start: startOffset,
				End:   endOffset,
				Label: buf.String,
			})
		}
		buf = nil
	}
	for _, token := range tokens {
		switch token.Label[0] {
		case 'O':
			flush()
			newTokens = append(newTokens, token)
		case 'B':
			flush()
			buf = &tokenizers.StringOffsetsPair{
				String: fmt.Sprintf("%s", token.Label[2:]), // copy
				Offsets: tokenizers.OffsetsType{
					Start: token.Start,
					End:   token.End,
				},
			}
		case 'I':
			if buf != nil {
				buf.Offsets.End = token.End
			} else { // same as 'B'
				buf = &tokenizers.StringOffsetsPair{
					String: fmt.Sprintf("%s", token.Label[2:]), // copy
					Offsets: tokenizers.OffsetsType{
						Start: token.Start,
						End:   token.End,
					},
				}
			}
		}
	}
	flush()
	return newTokens
}

func filterNotEntities(tokens []Token) []Token {
	ret := make([]Token, 0)
	for _, token := range tokens {
		if token.Label == "O" { // not an entity
			continue
		}
		ret = append(ret, token)
	}
	return ret
}
