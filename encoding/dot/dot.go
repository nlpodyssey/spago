// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dot allows exporting an encoding.Graph to Graphviz DOT format.
package dot

import (
	"io"
	"strconv"
	"text/template"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/encoding"
	"github.com/nlpodyssey/spago/mat"
)

// Encode encodes the graph g in Graphviz DOT format, writing the output to w.
func Encode[T mat.DType](g *encoding.Graph[T], w io.Writer) error {
	return dotTemplate.Execute(w, g)
}

// language=gotemplate
const templateText = `
{{- block "graph" . -}}
strict digraph {
	rankdir=LR;
	colorscheme="dark28";
	node [colorscheme="dark28"];
{{template "nodes" .}}
{{template "edges" .Edges}}
}
{{end -}}

{{- define "nodes" -}}
	{{- if .HasTimeStepHandler -}}
		{{- template "multiCluster" . -}}
	{{- else -}}
		{{- template "singleCluster" .NodesList -}}
	{{- end -}}
{{- end -}}

{{- define "singleCluster" -}}
	{{- range $i, $n := .}}
	{{printf 
		"%d [label=<<sup>%d</sup><br/><b>%s</b><br/><sub>%d×%d</sub>>,shape=%s]" 
		$i $i (.Name | html) .Value.Rows .Value.Columns (nodeShape $n)
	}}
	{{- end -}}
{{- end -}}

{{- define "multiCluster" -}}
	{{- range $timeStep, $nodeIDs := .NodesByTimeStep}}
	subgraph "cluster_timestep_{{$timeStep}}" {
		label="Time Step {{$timeStep}}";
		{{- $color := (timeStepColor $timeStep)}}
		color={{$color}};
		node [color={{$color}}];
		{{range $nodeIDs}}
			{{- $node := (index $.NodesList .)}}
		{{printf 
			"%d [label=<<sup>%d</sup><br/><b>%s</b><br/><sub>%d×%d</sub>>,shape=%s,color=%s]" 
			. . ($node.Name | html)
			$node.Value.Rows $node.Value.Columns (nodeShape $node) $color
		}}
		{{- end}}
	}
{{end -}}
{{- end -}}

{{- define "edges" -}}
	{{- range $fromNode, $toNodes := . -}}{{range $toNodes}}
	{{$fromNode}}->{{.}};
	{{- end -}}{{- end -}}
{{- end -}}
`

var (
	funcs = template.FuncMap{
		"timeStepColor": func(timeStep int) string {
			if timeStep < 0 {
				return `""`
			}
			// dark28 color schemes has colors from 1 to 8
			return strconv.Itoa((timeStep % 8) + 1)
		},
		"nodeShape": func(node any) string {
			switch node.(type) {
			case *ag.Operator[float32], *ag.Operator[float64]:
				return "oval"
			default:
				return "box"
			}
		},
	}
	dotTemplate = template.Must(
		template.New("DotTemplate").Funcs(funcs).Parse(templateText),
	)
)
