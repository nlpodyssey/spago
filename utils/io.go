// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import (
	"bufio"
	"bytes"
	"encoding/gob"
	"io"
	"log"
	"os"
)

// SerializeToFile serializes obj to file, using gob encoding.
func SerializeToFile(filename string, obj interface{}) (err error) {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = e
		}
	}()
	buf := bufio.NewWriter(f) // Buffered writing is essential to avoid memory leaks with large data
	err = gob.NewEncoder(buf).Encode(obj)
	if err != nil {
		return err
	}
	err = buf.Flush()
	if err != nil {
		return err
	}
	return nil
}

// DeserializeFromFile deserializes obj from file, using gob decoding.
func DeserializeFromFile(filename string, obj interface{}) (err error) {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = e
		}
	}()
	err = gob.NewDecoder(bufio.NewReader(f)).Decode(obj)
	if err != nil {
		return err
	}
	return
}

// CountLines efficiently counts the lines of text inside a file.
// See: https://stackoverflow.com/questions/24562942/golang-how-do-i-determine-the-number-of-lines-in-a-file-efficiently
func CountLines(filename string) (int, error) {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	buf := make([]byte, 32*1024)
	count := 0
	lineSep := []byte{'\n'}

	for {
		c, err := file.Read(buf)
		count += bytes.Count(buf[:c], lineSep)
		switch {
		case err == io.EOF:
			return count, nil
		case err != nil:
			return count, err
		}
	}
}
