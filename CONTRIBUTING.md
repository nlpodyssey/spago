# Contributing Guidelines

We're glad you're thinking about contributing to spaGO! If you think something is missing or could be improved, please open issues and pull requests. If you'd like to help this project grow, we'd love to have you! To start contributing, checking issues tagged as "good-first-issue" is a good start!

## Branching

The preferred flow is to fork the project, create branches in your fork, and submit PRs from your forked branch.

## Requirements

* [Go 1.14](https://golang.org/dl/)
* [Go Modules](https://blog.golang.org/using-go-modules)

## API Development

Install the following tools like this, if you haven't already.

```console
brew install protobuf
export GO111MODULE=on  # Enable module mode
go get github.com/golang/protobuf/protoc-gen-go
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc
```

After changing the gRPC protobuf specification, run `go generate ./...` from the top-level folder.
