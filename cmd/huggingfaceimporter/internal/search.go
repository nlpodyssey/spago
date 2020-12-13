// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// ModelCard is a model in the search results.
type ModelCard struct {
	// LastModified is the last time this model was modified.
	LastModified time.Time `json:"lastModified"`
	// ModelID is the model identifier
	ModelID string `json:"id"`
	// Author is the author's name.
	Author string `json:"author"`
	// Downloads is the number of downloads.
	Downloads int `json:"downloads"`
	// Tags is the list of tags.
	Tags []string `json:"tags"`
}

// LookupFromHuggingFace looks up a search query, empty query returns all.
func LookupFromHuggingFace(searchQuery string) (string, error) {
	loc, err := url.Parse("https://huggingface.co/models")
	if err != nil {
		return "", err
	}
	if searchQuery != "" {
		loc.RawQuery = url.Values{
			"search": []string{searchQuery},
		}.Encode()
	}

	resp, err := http.Get(loc.String())
	if err != nil {
		return "", err
	}

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	dataStr := string(data)
	startSubstr := "window.getAllModels = () => ["
	idx := strings.Index(dataStr, startSubstr)
	if idx < 0 {
		return "", errors.New("getAllModels not found")
	}
	idx += len(startSubstr) - 1
	dataStr = dataStr[idx:]
	endSubstr := "}\n];"
	endIdx := strings.Index(dataStr, endSubstr)
	dataStr = dataStr[:endIdx+len(endSubstr)-1]
	return dataStr, nil
}

// ParseSearchResults parses search results json.
func ParseSearchResults(dataJSON []byte) ([]*ModelCard, error) {
	var res []*ModelCard
	if err := json.Unmarshal(dataJSON, &res); err != nil {
		return nil, err
	}
	return res, nil
}
