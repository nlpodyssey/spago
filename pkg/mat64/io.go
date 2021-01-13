// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat64

import (
	"bytes"
	"encoding/binary"
	"errors"
	"github.com/nlpodyssey/spago/pkg/utils"
	"io"
	"math"
)

type header struct {
	Rows int64
	Cols int64
}

// maxLen is the biggest slice/array len one can create on a 32/64b platform.
const maxLen = int(^uint(0) >> 1)

var (
	headerSize    = binary.Size(header{})
	errTooBig     = errors.New("mat64: resulting data slice too big")
	errBadSize    = errors.New("mat64: invalid dimension")
	errZeroLength = errors.New("mat64: zero length in matrix dimension")
)

func (s header) marshalBinaryTo(w io.Writer) error {
	buf := bytes.NewBuffer(make([]byte, 0, headerSize))
	err := binary.Write(buf, binary.LittleEndian, s)
	if err != nil {
		return err
	}
	_, err = w.Write(buf.Bytes())
	return err
}

func (s *header) unmarshalBinary(buf []byte) error {
	err := binary.Read(bytes.NewReader(buf), binary.LittleEndian, s)
	if err != nil {
		return err
	}
	return nil
}

func (s *header) unmarshalBinaryFrom(r io.Reader) error {
	buf := make([]byte, headerSize)
	n, err := utils.ReadFull(r, buf)
	if err != nil {
		return err
	}
	return s.unmarshalBinary(buf[:n])
}

// MarshalBinaryTo encodes the receiver into a binary form and writes it into w.
func MarshalBinaryTo(m Matrix, w io.Writer) error {
	h := header{Rows: int64(m.Rows()), Cols: int64(m.Columns())}
	err := h.marshalBinaryTo(w)
	if err != nil {
		return err
	}
	var b [8]byte
	for _, num := range m.Data() {
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(num))
		_, err = w.Write(b[:])
		if err != nil {
			return err
		}
	}
	return nil
}

// NewUnmarshalBinaryFrom decodes binary data form the reader and returns a new Dense matrix.
func NewUnmarshalBinaryFrom(r io.Reader) (*Dense, error) {
	var h header
	err := h.unmarshalBinaryFrom(r)
	if err != nil {
		return nil, err
	}
	rows := int(h.Rows)
	cols := int(h.Cols)
	if rows < 0 || cols < 0 {
		return nil, errBadSize
	}
	size := rows * cols
	if size == 0 {
		return nil, errZeroLength
	}
	if size < 0 || size > maxLen {
		return nil, errTooBig
	}

	data := make([]Float, rows*cols)
	var b [8]byte
	for i := range data {
		_, err = utils.ReadFull(r, b[:])
		if err != nil {
			if err == io.EOF {
				return nil, io.ErrUnexpectedEOF
			}
			return nil, err
		}
		data[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[:]))
	}

	m := NewDense(rows, cols, data)
	return m, nil
}

// UnmarshalBinaryFrom decodes binary data form the reader into the given Matrix.
func UnmarshalBinaryFrom(m Matrix, r io.Reader) error {
	var h header
	err := h.unmarshalBinaryFrom(r)
	if err != nil {
		return err
	}
	rows := int(h.Rows)
	cols := int(h.Cols)
	if rows < 0 || cols < 0 {
		return errBadSize
	}
	size := rows * cols
	if size == 0 {
		return errZeroLength
	}
	if size < 0 || size > maxLen {
		return errTooBig
	}
	if rows != m.Rows() || cols != m.Columns() {
		return errBadSize
	}
	var b [8]byte
	data := m.Data()
	for i := range data {
		_, err = utils.ReadFull(r, b[:])
		if err != nil {
			if err == io.EOF {
				return io.ErrUnexpectedEOF
			}
			return err
		}
		data[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[:]))
	}
	return nil
}

// MarshalBinarySlice marshals a slice of matrices to binary.
func MarshalBinarySlice(ms []Matrix, w io.Writer) error {
	for _, m := range ms {
		err := MarshalBinaryTo(m, w)
		if err != nil {
			return err
		}
	}
	return nil
}

// UnmarshalBinarySlice unmarshals a slice of matrices from binary.
func UnmarshalBinarySlice(ms []Matrix, r io.Reader) error {
	for _, m := range ms {
		err := UnmarshalBinaryFrom(m, r)
		if err != nil {
			return err
		}
	}
	return nil
}

// NewUnmarshalBinarySlice decodes binary data form the reader into the given slice of Matrix.
func NewUnmarshalBinarySlice(ms []Matrix, r io.Reader) error {
	for i := range ms {
		m, err := NewUnmarshalBinaryFrom(r)
		ms[i] = m
		if err != nil {
			return err
		}
	}
	return nil
}
