// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httputils

import (
	"log"
	"net/http"

	"github.com/nlpodyssey/spago/pkg/utils/httphandlers"
)

// RunHTTPServer listens on the given address and serves the given mux using HTTP
// (optionally over TLS), and blocks until done.
func RunHTTPServer(address string, tlsDisable bool, tlsCert, tlsKey string, mux *http.ServeMux) {
	if tlsDisable {
		log.Fatal(http.ListenAndServe(address, newRecoveryHandler(mux)))
	} else {
		log.Fatal(http.ListenAndServeTLS(address, tlsCert, tlsKey, newRecoveryHandler(mux)))
	}
}

func newRecoveryHandler(r *http.ServeMux) http.Handler {
	return httphandlers.RecoveryHandler(httphandlers.PrintRecoveryStack(true))(r)
}
