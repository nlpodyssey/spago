// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package corpora

import (
	"archive/tar"
	"bufio"
	"compress/gzip"
	"io"
	"log"
	"os"
)

type TextCorpusIterator interface {
	ForEachLine(callback func(i int, line string))
}

var _ TextCorpusIterator = &GZipCorpusIterator{}

type GZipCorpusIterator struct {
	CorpusPath string
}

// NewGZipCorpusIterator returns a new GZipCorpusIterator.
func NewGZipCorpusIterator(corpusPath string) *GZipCorpusIterator {
	return &GZipCorpusIterator{CorpusPath: corpusPath}
}

func (c *GZipCorpusIterator) ForEachLine(callback func(i int, text string)) {
	f, err := os.Open(c.CorpusPath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	uncompressedStream, err := gzip.NewReader(f)
	if err != nil {
		log.Fatal(err)
	}
	tarReader := tar.NewReader(uncompressedStream)
	i := 0
	for true {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("Next() failed: %s", err.Error())
		}
		if header.Typeflag == tar.TypeReg {
			scanner := bufio.NewScanner(tarReader)
			for scanner.Scan() {
				i++
				callback(i, scanner.Text())
			}
			if err := scanner.Err(); err != nil {
				log.Fatal(err)
			}
		}
	}
}
