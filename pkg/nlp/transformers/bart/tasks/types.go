package tasks

import mat "github.com/nlpodyssey/spago/pkg/mat32"

// ClassConfidencePair is a JSON-serializable pair of Class and Confidence.
type ClassConfidencePair struct {
	Class      string    `json:"class"`
	Confidence mat.Float `json:"confidence"`
}

// ClassifyResponse is a JSON-serializable structure which holds server
// classification response data.
type ClassifyResponse struct {
	Class        string                `json:"class"`
	Confidence   mat.Float             `json:"confidence"`
	Distribution []ClassConfidencePair `json:"distribution"`

	// Took is the number of milliseconds it took the server to execute the request.
	Took int64 `json:"took"`
}
