// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEncodeKey(t *testing.T) {
	stringTestCases := []struct {
		input    string
		expected []byte
	}{
		{"", []byte{}},
		{"x", []byte{'x'}},
		{"Foo", []byte{'F', 'o', 'o'}},
	}
	for _, tc := range stringTestCases {
		t.Run(fmt.Sprintf("%T %#v", tc.input, tc.input), func(t *testing.T) {
			assert.Equal(t, tc.expected, encodeKey(tc.input))
		})
	}

	bytesTestCases := [][]byte{
		nil,
		{},
		{42},
		{0xCA, 0xFE},
	}
	for _, tc := range bytesTestCases {
		t.Run(fmt.Sprintf("%T %#v", tc, tc), func(t *testing.T) {
			assert.Equal(t, tc, encodeKey(tc))
		})
	}

	intTestCases := []struct {
		input    int
		expected []byte
	}{
		{0, []byte{0, 0, 0, 0, 0, 0, 0, 0}},
		{0xcafe, []byte{0xfe, 0xca, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
		{0xdeadbeef, []byte{0xef, 0xbe, 0xad, 0xde, 0x00, 0x00, 0x00, 0x00}},
		{0x1122334455667788, []byte{0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11}},
	}
	for _, tc := range intTestCases {
		t.Run(fmt.Sprintf("%T %#v", tc.input, tc.input), func(t *testing.T) {
			assert.Equal(t, tc.expected, encodeKey(tc.input))
		})
	}
}

func TestStringifyKey(t *testing.T) {
	stringTestCases := []string{
		"",
		"Foo",
	}
	for _, tc := range stringTestCases {
		t.Run(fmt.Sprintf("%T %#v", tc, tc), func(t *testing.T) {
			s := stringifyKey(tc)
			assert.Equal(t, tc, s)
		})
	}

	bytesTestCases := []struct {
		input    []byte
		expected string
	}{
		{nil, ""},
		{[]byte{}, ""},
		{[]byte{'a'}, "a"},
		{[]byte{'F', 'o', 'o'}, "Foo"},
	}
	for _, tc := range bytesTestCases {
		t.Run(fmt.Sprintf("%T %#v", tc.input, tc.input), func(t *testing.T) {
			s := stringifyKey(tc.input)
			assert.Equal(t, tc.expected, s)
		})
	}

	intTestCases := []struct {
		input    int
		expected string
	}{
		{0, "\x00\x00\x00\x00\x00\x00\x00\x00"},
		{0xcafe, "\xfe\xca\x00\x00\x00\x00\x00\x00"},
		{0xdeadbeef, "\xef\xbe\xad\xde\x00\x00\x00\x00"},
		{0x1122334455667788, "\x88\x77\x66\x55\x44\x33\x22\x11"},
	}
	for _, tc := range intTestCases {
		t.Run(fmt.Sprintf("%T %#v", tc.input, tc.input), func(t *testing.T) {
			s := stringifyKey(tc.input)
			assert.Equal(t, tc.expected, s)
		})
	}
}
