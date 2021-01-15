// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package grpcutils

import (
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"log"
	"net"
	"time"
)

// GRPCServerConfig provides server configuration parameters for creating
// a GRPC server (see NewGRPCServer).
type GRPCServerConfig struct {
	TLSDisable      bool
	TLSCert         string
	TLSKey          string
	TimeoutSeconds  int
	MaxRequestBytes int
}

// NewGRPCServer returns grpc.Server objects, optionally configured for TLS.
func NewGRPCServer(config GRPCServerConfig) *grpc.Server {
	serverOptions := createServerOptions(config)
	return grpc.NewServer(serverOptions...)
}

func createServerOptions(config GRPCServerConfig) []grpc.ServerOption {
	options := []grpc.ServerOption{
		grpc.MaxRecvMsgSize(config.MaxRequestBytes),
		// ConnectionTimeout is EXPERIMENTAL and may be changed or removed in a later release.
		grpc.ConnectionTimeout(time.Duration(config.TimeoutSeconds) * time.Second),
	}

	if !config.TLSDisable {
		creds, err := credentials.NewServerTLSFromFile(config.TLSCert, config.TLSKey)
		if err != nil {
			log.Fatalf("failed to read TLS certs: %v\n", err)
		}
		options = append(options, grpc.Creds(creds))
	}

	return options
}

// RunGRPCServer listens on the given address and serves the given *grpc.Server,
// and blocks until done.
func RunGRPCServer(grpcAddress string, grpcServer *grpc.Server) {
	listener := newListenerForGRPC(grpcAddress)
	log.Fatal(grpcServer.Serve(listener))
}

func newListenerForGRPC(grpcAddress string) net.Listener {
	result, err := net.Listen("tcp", grpcAddress)

	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	return result
}
