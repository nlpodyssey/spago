// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

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
	errTooBig     = errors.New("mat32: resulting data slice too big")
	errBadSize    = errors.New("mat32: invalid dimension")
	errZeroLength = errors.New("mat32: zero length in matrix dimension")
)

func (s header) marshalBinaryTo(w io.Writer) (int, error) {
	buf := bytes.NewBuffer(make([]byte, 0, headerSize))
	err := binary.Write(buf, binary.LittleEndian, s)
	if err != nil {
		return 0, err
	}
	return w.Write(buf.Bytes())
}

func (s *header) unmarshalBinary(buf []byte) error {
	err := binary.Read(bytes.NewReader(buf), binary.LittleEndian, s)
	if err != nil {
		return err
	}
	return nil
}

func (s *header) unmarshalBinaryFrom(r io.Reader) (int, error) {
	buf := make([]byte, headerSize)
	n, err := utils.ReadFull(r, buf)
	if err != nil {
		return n, err
	}
	return n, s.unmarshalBinary(buf[:n])
}

// MarshalBinaryTo encodes the receiver into a binary form and writes it into w.
// MarshalBinaryTo returns the number of bytes written into w and an error, if any.
func MarshalBinaryTo(m Matrix, w io.Writer) (int, error) {
	h := header{Rows: int64(m.Rows()), Cols: int64(m.Columns())}
	n, err := h.marshalBinaryTo(w)
	if err != nil {
		return n, err
	}
	var b [8]byte
	for _, num := range m.Data() {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(num))
		nn, err := w.Write(b[:])
		n += nn
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

// NewUnmarshalBinaryFrom decodes binary data form the reader and returns a new
// Dense matrix, along with the number of bytes read and an error, if any.
func NewUnmarshalBinaryFrom(r io.Reader) (*Dense, int, error) {
	var h header
	n, err := h.unmarshalBinaryFrom(r)
	if err != nil {
		return nil, n, err
	}
	rows := int(h.Rows)
	cols := int(h.Cols)
	if rows < 0 || cols < 0 {
		return nil, n, errBadSize
	}
	size := rows * cols
	if size == 0 {
		return nil, n, errZeroLength
	}
	if size < 0 || size > maxLen {
		return nil, n, errTooBig
	}

	data := make([]Float, rows*cols)
	var b [8]byte
	for i := range data {
		nn, err := utils.ReadFull(r, b[:])
		n += nn
		if err != nil {
			if err == io.EOF {
				return nil, n, io.ErrUnexpectedEOF
			}
			return nil, n, err
		}
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[:]))
	}

	m := NewDense(rows, cols, data)
	return m, n, nil
}

// UnmarshalBinaryFrom decodes binary data form the reader into the given Matrix,
// and returns the number of bytes read and an error, if any.
func UnmarshalBinaryFrom(m Matrix, r io.Reader) (int, error) {
	var h header
	n, err := h.unmarshalBinaryFrom(r)
	if err != nil {
		return n, err
	}
	rows := int(h.Rows)
	cols := int(h.Cols)
	if rows < 0 || cols < 0 {
		return n, errBadSize
	}
	size := rows * cols
	if size == 0 {
		return n, errZeroLength
	}
	if size < 0 || size > maxLen {
		return n, errTooBig
	}
	if rows != m.Rows() || cols != m.Columns() {
		return n, errBadSize
	}
	var b [8]byte
	data := m.Data()
	for i := range data {
		nn, err := utils.ReadFull(r, b[:])
		n += nn
		if err != nil {
			if err == io.EOF {
				return n, io.ErrUnexpectedEOF
			}
			return n, err
		}
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[:]))
	}
	return n, nil
}

// MarshalBinarySlice returns the number of bytes written into w and an error, if any.
func MarshalBinarySlice(ms []Matrix, w io.Writer) (int, error) {
	n := 0
	for _, m := range ms {
		num, err := MarshalBinaryTo(m, w)
		n += num
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

// UnmarshalBinarySlice decodes the binary form the reader into the slice itself and returns
// the number of bytes read and an error if any.
func UnmarshalBinarySlice(ms []Matrix, r io.Reader) (int, error) {
	n := 0
	for _, m := range ms {
		num, err := UnmarshalBinaryFrom(m, r)
		n += num
		if err != nil {
			return n, err
		}
	}
	return n, nil
}

// NewUnmarshalBinarySlice decodes binary data form the reader into the given slice
// of Matrix, and returns the number of bytes read and an error, if any.
func NewUnmarshalBinarySlice(ms []Matrix, r io.Reader) (int, error) {
	n := 0
	for i := range ms {
		m, num, err := NewUnmarshalBinaryFrom(r)
		ms[i] = m
		n += num
		if err != nil {
			return n, err
		}
	}
	return n, nil
}
