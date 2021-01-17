// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httputils

import (
	"github.com/nlpodyssey/spago/pkg/utils/httphandlers"
	"log"
	"net/http"
	"time"
)

// DefaultTimeoutSeconds is a sensible default value for HTTPServerConfig.TimeoutSeconds.
const DefaultTimeoutSeconds = 60

// DefaultMaxRequestBytes is a sensible default value for the maximum number of
// bytes the server will read parsing the request's body.
const DefaultMaxRequestBytes = 1 << 20 // 1 MB

// HTTPServerConfig provides server configuration parameters for running
// an HTTP server (see RunHTTPServer).
type HTTPServerConfig struct {
	Address         string
	TLSDisable      bool
	TLSCert         string
	TLSKey          string
	TimeoutSeconds  int
	MaxRequestBytes int
}

// RunHTTPServer listens on the given address and serves the given mux using HTTP
// (optionally over TLS), and blocks until done.
func RunHTTPServer(config HTTPServerConfig, h http.Handler) {
	timeout := time.Duration(config.TimeoutSeconds) * time.Second
	server := &http.Server{
		Addr: config.Address,
		Handler: http.TimeoutHandler(
			newRecoveryHandler(&maxRequestBytesHandler{
				h: h,
				n: int64(config.MaxRequestBytes),
			}),
			timeout,
			"Timeout",
		),
	}

	var err error
	if config.TLSDisable {
		err = server.ListenAndServe()
	} else {
		err = server.ListenAndServeTLS(config.TLSCert, config.TLSKey)
	}
	log.Fatal(err)
}

type maxRequestBytesHandler struct {
	h http.Handler
	n int64
}

func (h *maxRequestBytesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, h.n)
	h.h.ServeHTTP(w, r)
}

func newRecoveryHandler(h http.Handler) http.Handler {
	return httphandlers.RecoveryHandler(httphandlers.PrintRecoveryStack(true))(h)
}
