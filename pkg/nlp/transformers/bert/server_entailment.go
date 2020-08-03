// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/json"
	"net/http"
	"sort"
	"time"

	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
)

type TEBody struct {
	Premise    string `json:"premise"`
	Hypothesis string `json:"hypothesis"`
}

func (s *Server) TextualEntailmentHandler(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*") // that's intended for testing purposes only
	w.Header().Set("Content-Type", "application/json")

	var body TEBody
	err := json.NewDecoder(req.Body).Decode(&body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	result := s.textualEntailment(body.Premise, body.Hypothesis)
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

// TODO: This method is too long; it needs to be refactored.
func (s *Server) textualEntailment(premise string, hypothesis string) *ClassifyResponse {
	start := time.Now()

	tokenizer := wordpiecetokenizer.New(s.model.Vocabulary)
	origQuestionTokens := tokenizer.Tokenize(premise)
	origPassageTokens := tokenizer.Tokenize(hypothesis)

	cls := wordpiecetokenizer.DefaultClassToken
	sep := wordpiecetokenizer.DefaultSequenceSeparator
	tokenized := append([]string{cls}, append(tokenizers.GetStrings(origQuestionTokens), sep)...)
	tokenized = append(tokenized, append(tokenizers.GetStrings(origPassageTokens), sep)...)

	g := ag.NewGraph()
	defer g.Clear()
	proc := s.model.NewProc(g).(*Processor)
	proc.SetMode(nn.Inference)
	encoded := proc.Encode(tokenized)

	logits := proc.SequenceClassification(encoded)
	probs := f64utils.SoftMax(logits.Value().Data())
	best := f64utils.ArgMax(probs)
	class := s.model.Classifier.Config.Labels[best]

	distribution := make([]ClassConfidencePair, len(probs))
	for i := 0; i < len(probs); i++ {
		distribution[i] = ClassConfidencePair{
			Class:      s.model.Classifier.Config.Labels[i],
			Confidence: probs[i],
		}
	}

	sort.Slice(distribution, func(i, j int) bool {
		return distribution[i].Confidence > distribution[j].Confidence
	})

	return &ClassifyResponse{
		Class:        class,
		Confidence:   probs[best],
		Distribution: distribution,
		Took:         time.Since(start).Milliseconds(),
	}
}
