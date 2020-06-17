package main

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
	ModelID string `json:"modelId"`
	// Author is the author's name.
	Author string `json:"author"`
	// Downloads is the number of downloads.
	Downloads int `json:"downloads"`
	// Tags is the list of tags.
	Tags []string `json:"tags"`
	// CardSource is the source of the search card.
	CardSource string `json:"cardSource"`
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
	startSubstr := "window.getModelCards = () => ["
	idx := strings.Index(dataStr, startSubstr)
	if idx < 0 {
		return "", errors.New("getModelCards not found")
	}
	idx += len(startSubstr) - 1
	dataStr = dataStr[idx:]
	endSubstr := "}\n];"
	endIdx := strings.Index(dataStr, endSubstr)
	dataStr = dataStr[:endIdx+len(endSubstr)-1]
	return dataStr, nil
}

// ParseSearchResults parses search results json.
func ParseSearchResults(dataJson []byte) ([]*ModelCard, error) {
	var res []*ModelCard
	if err := json.Unmarshal(dataJson, &res); err != nil {
		return nil, err
	}
	return res, nil
}
