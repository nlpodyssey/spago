// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"context"
	"encoding/json"
	"fmt"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bert/grpcapi"
	"net/http"
	"time"
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

	result, err := s.encode(body.Text, body.PoolingStrategy)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
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
	result, err := s.encode(req.GetText(), req.GetPoolingStrategy())
	if err != nil {
		return nil, err
	}

	vector32 := make([]float32, len(result.Data))
	for i, num := range result.Data {
		vector32[i] = float32(num)
	}

	return &grpcapi.EncodeReply{
		Vector: vector32,
		Took:   result.Took,
	}, nil
}

func (s *Server) encode(text string, poolingStrategy grpcapi.EncodeRequest_PoolingStrategy) (*EncodeResponse, error) {
	start := time.Now()
	ps, err := getPoolingStrategyFromEncodeRequest(poolingStrategy)
	if err != nil {
		return nil, err
	}
	encoded, err := s.model.Vectorize(text, ps)
	if err != nil {
		return nil, err
	}
	return &EncodeResponse{
		Data: encoded.Data(),
		Took: time.Since(start).Milliseconds(),
	}, nil
}

func getPoolingStrategyFromEncodeRequest(poolingStrategy grpcapi.EncodeRequest_PoolingStrategy) (PoolingStrategy, error) {
	switch poolingStrategy {
	case grpcapi.EncodeRequest_REDUCE_MEAN:
		return ReduceMean, nil
	case grpcapi.EncodeRequest_REDUCE_MAX:
		return ReduceMax, nil
	case grpcapi.EncodeRequest_REDUCE_MEAN_MAX:
		return ReduceMeanMax, nil
	case grpcapi.EncodeRequest_CLS_TOKEN:
		return ClsToken, nil
	default:
		return -1, fmt.Errorf("bert: invalid pooling strategy")
	}
}
