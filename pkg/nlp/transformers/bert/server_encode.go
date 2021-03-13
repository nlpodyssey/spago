// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"context"
	"encoding/json"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
	"net/http"
	"runtime"
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

	result := s.encode(body.Text, body.PoolingStrategy)
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

// EncodeResponse is a JSON-serializable server response for BERT "encode" requests.
type EncodeResponse struct {
	Data []mat.Float `json:"data"`
	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
}

// Encode handles an encoding request over gRPC.
// TODO(evanmcclure@gmail.com) Reuse the gRPC message type for HTTP requests.
func (s *Server) Encode(_ context.Context, req *grpcapi.EncodeRequest) (*grpcapi.EncodeReply, error) {
	result := s.encode(req.GetText(), req.GetPoolingStrategy())

	vector32 := make([]float32, len(result.Data))
	for i, num := range result.Data {
		vector32[i] = float32(num)
	}

	return &grpcapi.EncodeReply{
		Vector: vector32,
		Took:   result.Took,
	}, nil
}

func (s *Server) encode(text string, poolingStrategy grpcapi.EncodeRequest_PoolingStrategy) *EncodeResponse {
	start := time.Now()
	tokenizer := wordpiecetokenizer.New(s.model.Vocabulary)
	origTokens := tokenizer.Tokenize(text)
	tokenized := pad(tokenizers.GetStrings(origTokens))

	g := ag.NewGraph(ag.ConcurrentComputations(runtime.NumCPU()))
	defer g.Clear()
	proc := nn.Reify(nn.Context{Graph: g, Mode: nn.Inference}, s.model).(*Model)
	encoded := proc.Encode(tokenized)

	var pooled ag.Node
	switch poolingStrategy {
	case grpcapi.EncodeRequest_REDUCE_MEAN:
		pooled = g.Mean(encoded)
	case grpcapi.EncodeRequest_REDUCE_MAX:
		pooled = Max(g, encoded)
	case grpcapi.EncodeRequest_REDUCE_MEAN_MAX:
		pooled = g.Concat(g.Mean(encoded), Max(g, encoded))
	case grpcapi.EncodeRequest_CLS_TOKEN:
		pooled = proc.Pool(encoded)
	default:
		panic("bert: invalid pooling strategy")
	}

	return &EncodeResponse{
		Data: pooled.Value().Data(),
		Took: time.Since(start).Milliseconds(),
	}
}

// Mean returns the value that describes the average of the sample.
func Max(g *ag.Graph, xs []ag.Node) ag.Node {
	maxVector := xs[0]
	for i := 1; i < len(xs); i++ {
		maxVector = g.Max(maxVector, xs[i])
	}
	return maxVector
}
