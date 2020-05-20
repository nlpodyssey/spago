// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vocabulary

import (
	"bufio"
	"os"
	"sync/atomic"
)

// Provider is an interface for exposing a vocabulary.
type Provider interface {
	Vocabulary() *Vocabulary
}

type Vocabulary struct {
	maxId   int64
	terms   map[string]int
	inverse []string
}

// New returns a new vocabulary populated with the terms.
func New(terms []string) *Vocabulary {
	c := &Vocabulary{
		maxId:   -1,
		terms:   make(map[string]int),
		inverse: make([]string, 0),
	}
	for _, w := range terms {
		c.Add(w)
	}
	return c
}

// NewFromFile returns a new vocabulary populated with the content of a file.
func NewFromFile(path string) (*Vocabulary, error) {
	f, err := os.Open(path)
	if err != nil {
		return &Vocabulary{}, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	voc := New([]string{})
	for scanner.Scan() {
		voc.Add(scanner.Text())
	}
	return voc, nil
}

// Items returns all items.
func (c *Vocabulary) Items() []string {
	return c.inverse
}

// ID returns the ID of a term and whether or not it was found in the vocabulary.
func (c *Vocabulary) Id(term string) (int, bool) {
	id, ok := c.terms[term]
	return id, ok
}

// MustID returns the ID of a term.
// It panics if the term is not in the vocabulary.
func (c *Vocabulary) MustId(term string) int {
	id, ok := c.Id(term)
	if !ok {
		panic("vocabulary: term not found.")
	}
	return id
}

// Term returns the term given the ID, and whether or not it was found in the vocabulary.
func (c *Vocabulary) Term(id int) (string, bool) {
	size := atomic.LoadInt64(&c.maxId)
	maxId := int(size)
	if id >= maxId {
		return "", false
	}
	return c.inverse[id], true
}

// MustTerm returns the term given the ID.
// It panics if the term is not in the vocabulary.
func (c *Vocabulary) MustTerm(id int) string {
	term, ok := c.Term(id)
	if !ok {
		panic("vocabulary: id not found.")
	}
	return term
}

// Add adds a term to the vocabulary and returns its ID.
// If a term was already in the corpus, it just returns the ID.
func (c *Vocabulary) Add(term string) int {
	if id, ok := c.terms[term]; ok {
		return id
	}
	id := atomic.AddInt64(&c.maxId, 1)
	c.terms[term] = int(id)
	c.inverse = append(c.inverse, term)
	return int(id)
}

// Size returns the size of the vocabulary.
func (c *Vocabulary) Size() int {
	size := atomic.LoadInt64(&c.maxId)
	return int(size)
}

// LongestPrefix returns the longest term in the vocabulary that is the prefix of the input term
func (c *Vocabulary) LongestPrefix(term string) string {
	for i := len(term); i > 0; i-- {
		sub := term[:i]
		if _, ok := c.terms[sub]; ok {
			return sub
		}
	}
	return ""
}
