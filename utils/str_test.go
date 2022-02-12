// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestBeforeSpace(t *testing.T) {
	t.Parallel()
	examples := []struct{ s, expected string }{
		{"", ""},
		{" ", ""},
		{"foo", ""},
		{" foo", ""},
		{"  foo", ""},
		{"x  foo", "x"},
		{"foo bar", "foo"},
		{"foo  bar", "foo"},
		{"foo bar baz", "foo"},
	}
	for _, ex := range examples {
		t.Run(fmt.Sprintf("%#v", ex.s), func(t *testing.T) {
			assert.Equal(t, ex.expected, BeforeSpace(ex.s))
		})
	}
}

func TestAfterSpace(t *testing.T) {
	t.Parallel()
	examples := []struct{ s, expected string }{
		{"", ""},
		{" ", ""},
		{"foo", ""},
		{"foo ", ""},
		{"foo  ", " "},
		{"foo x", "x"},
		{"foo  x", " x"},
		{"foo bar", "bar"},
		{"foo  bar", " bar"},
		{"foo bar baz", "bar baz"},
	}
	for _, ex := range examples {
		t.Run(fmt.Sprintf("%#v", ex.s), func(t *testing.T) {
			assert.Equal(t, ex.expected, AfterSpace(ex.s))
		})
	}
}

func TestBefore(t *testing.T) {
	t.Parallel()
	examples := []struct{ s, expected string }{
		{"", ""},
		{"!?", ""},
		{"foo", ""},
		{"!?foo", ""},
		{"!?!?foo", ""},
		{"x!?!?foo", "x"},
		{"foo!?bar", "foo"},
		{"foo!?!?bar", "foo"},
		{"foo!?bar!?baz", "foo"},
	}
	for _, ex := range examples {
		t.Run(fmt.Sprintf("%#v", ex.s), func(t *testing.T) {
			assert.Equal(t, ex.expected, Before(ex.s, "!?"))
		})
	}
}

func TestAfter(t *testing.T) {
	t.Parallel()
	examples := []struct{ s, expected string }{
		{"", ""},
		{"!?", ""},
		{"foo", ""},
		{"foo!?", ""},
		{"foo!?!?", "!?"},
		{"foo!?x", "x"},
		{"foo!?!?x", "!?x"},
		{"foo!?bar", "bar"},
		{"foo!?!?bar", "!?bar"},
		{"foo!?bar!?baz", "bar!?baz"},
	}
	for _, ex := range examples {
		t.Run(fmt.Sprintf("%#v", ex.s), func(t *testing.T) {
			assert.Equal(t, ex.expected, After(ex.s, "!?"))
		})
	}
}

func TestSplitByRune(t *testing.T) {
	t.Parallel()
	examples := []struct {
		s        string
		expected []string
	}{
		{"", []string{}},
		{"x", []string{"x"}},
		{"bar", []string{"b", "a", "r"}},
		{"bar", []string{"b", "a", "r"}},
		{"ü", []string{"ü"}},
		{"süß", []string{"s", "ü", "ß"}},
	}
	for _, ex := range examples {
		t.Run(fmt.Sprintf("%#v", ex.s), func(t *testing.T) {
			assert.Equal(t, ex.expected, SplitByRune(ex.s))
		})
	}
}

func TestReverseString(t *testing.T) {
	t.Parallel()
	examples := []struct{ s, expected string }{
		{"", ""},
		{"x", "x"},
		{"xy", "yx"},
		{"bar", "rab"},
		{"foo bar", "rab oof"},
	}
	for _, ex := range examples {
		t.Run(fmt.Sprintf("%#v", ex.s), func(t *testing.T) {
			assert.Equal(t, ex.expected, ReverseString(ex.s))
		})
	}
}
