// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sequencelabeler

//go:generate protoc --go_out=Mgrpc/service_config/service_config.proto=/internal/proto/grpc_service_config:.  --go-grpc_out=Mgrpc/service_config/service_config.proto=/internal/proto/grpc_service_config:. --go_opt=paths=source_relative --go-grpc_opt=paths=source_relative grpcapi/sequencelabeler.proto

import (
	"net/http"

	"github.com/nlpodyssey/spago/pkg/nlp/sequencelabeler/grpcapi"
	"github.com/nlpodyssey/spago/pkg/utils/grpcutils"
	"github.com/nlpodyssey/spago/pkg/utils/httputils"
	"github.com/nlpodyssey/spago/pkg/webui/ner"
)

type Server struct {
	model *Model

	// UnimplementedSequenceLabelerServer must be embedded to have forward compatible implementations for gRPC.
	grpcapi.UnimplementedSequenceLabelerServer
}

func NewServer(model *Model) *Server {
	return &Server{
		model: model,
	}
}

func (s *Server) Start(address, grpcAddress, tlsCert, tlsKey string, tlsDisable bool) {
	mux := http.NewServeMux()
	mux.HandleFunc("/ner-ui", ner.Handler)
	mux.HandleFunc("/analyze", s.analyze)

	go httputils.RunHTTPServer(address, tlsDisable, tlsCert, tlsKey, mux)

	grpcServer := grpcutils.NewGRPCServer(tlsDisable, tlsCert, tlsKey)
	grpcapi.RegisterSequenceLabelerServer(grpcServer, s)
	grpcutils.RunGRPCServer(grpcAddress, grpcServer)
}
