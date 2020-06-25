// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"syscall"
)

const (
	help = `
spaGO is a beautiful and maintainable machine learning library written in Go designed to support relevant neural network architectures in natural language processing tasks.

Usage:
  <command> [arguments]"

The commands are:

   bert_server             demo server for question answering
   hugging_face_importer   demo program for model importing
   ner-server              demo server for named entities recognition

   See README.md for more information about run the demo servers using docker.
`
)

// The script docker-entrypoint.sh wraps access to the demo
// programs for named entities recognition (ner-server), model
// importing (hugging_face_importer), and question answering
// (bert_server).
func main() {

	// The help screen is printed to the user when no commands
	// are given, or when the command "help" is given.
	if len(os.Args) == 1 || strings.EqualFold(os.Args[1], "help") {
		fmt.Println(help)
		os.Exit(0)
	}

	// Run the commands defined by the Dockerfile CMD directive or overrides.
	proc := os.Args[1]
	_, err := exec.LookPath(proc)
	if err != nil {
		panic(err)
	}

	args := os.Args[1:]
	fmt.Printf("Running command: '%s'\n", strings.Join(args, " "))
	if err := syscall.Exec(proc, args, os.Environ()); err != nil {
		panic(err)
	}
}
