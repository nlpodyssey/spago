// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import (
	"bufio"
	"bytes"
	"io"
	"log"
	"os"
)

// Serializer is implemented by any value that has the Serialize method.
type Serializer interface {
	Serialize(w io.Writer) (int, error)
}

// Deserializer is implemented by any value that has the Deserialize method.
type Deserializer interface {
	Deserialize(r io.Reader) (int, error)
}

// SerializeToFile serializes obj to file.
func SerializeToFile(filename string, obj Serializer) (err error) {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	buf := bufio.NewWriter(f) // Buffered writing is essential to avoid memory leaks with large data
	_, err = obj.Serialize(buf)
	if err != nil {
		return err
	}
	err = buf.Flush()
	if err != nil {
		return err
	}
	return
}

// DeserializeFromFile deserializes obj from file.
func DeserializeFromFile(filename string, obj Deserializer) (err error) {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = obj.Deserialize(bufio.NewReader(f))
	if err != nil {
		return err
	}
	return
}

// ReadFull reads from r into buf until it has read len(buf).
// It returns the number of bytes copied and an error if fewer bytes were read.
// If an EOF happens after reading fewer than len(buf) bytes, io.ErrUnexpectedEOF is returned.
func ReadFull(r io.Reader, buf []byte) (int, error) {
	var n int
	var err error
	for n < len(buf) && err == nil {
		var nn int
		nn, err = r.Read(buf[n:])
		n += nn
	}
	if n == len(buf) {
		return n, nil
	}
	if err == io.EOF {
		return n, io.ErrUnexpectedEOF
	}
	return n, err
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
