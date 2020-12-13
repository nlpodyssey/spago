// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartconfig

import (
	"encoding/json"
	"os"
)

const (
	DefaultConfigurationFile = "config.json"
	DefaultModelFile         = "spago_model.bin"
	DefaultEmbeddingsStorage = "embeddings_storage"
)

// Config contains the global configuration of the BART model and the heads of fine-tuning tasks.
// The configuration coincides with that of Hugging Face to facilitate compatibility between the two architectures.
type Config struct {
	NumLabels                  int               `json:"_num_labels"`
	ActivationDropout          float64           `json:"activation_dropout"`
	ActivationFunction         string            `json:"activation_function"`
	BiasLogits                 bool              `json:"add_bias_logits"`
	FinalLayerNorm             bool              `json:"add_final_layer_norm"`
	Architecture               []string          `json:"architectures"`
	AttentionDropout           float64           `json:"attention_dropout"`
	BosTokenID                 int               `json:"bos_token_id"`
	ClassifierDropout          float64           `json:"classif_dropout"`
	DModel                     int               `json:"d_model"`
	DecoderAttentionHeads      int               `json:"decoder_attention_heads"`
	DecoderFFNDim              int               `json:"decoder_ffn_dim"`
	DecoderLayerDrop           float64           `json:"decoder_layerdrop"`
	DecoderLayers              int               `json:"decoder_layers"`
	Dropout                    float64           `json:"dropout"`
	EncoderAttentionHeads      int               `json:"encoder_attention_heads"`
	EncoderFFNDim              int               `json:"encoder_ffn_dim"`
	EncoderLayerDrop           float64           `json:"encoder_layerdrop"`
	EncoderLayers              int               `json:"encoder_layers"`
	EosTokenID                 int               `json:"eos_token_id"`
	ExtraPosEmbedding          int               `json:"extra_pos_embeddings"`
	FineTuningTask             string            `json:"finetuning_task"`
	ForceBosTokenToBeGenerated bool              `json:"force_bos_token_to_be_generated"`
	ID2Label                   map[string]string `json:"id2label"`
	InitStd                    float64           `json:"init_std"`
	IsEncoderDecoder           bool              `json:"is_encoder_decoder"`
	Label2ID                   map[string]int    `json:"label2id"`
	MaxPositionEmbeddings      int               `json:"max_position_embeddings"`
	ModelType                  string            `json:"model_type"`
	NormalizeBefore            bool              `json:"normalize_before"`
	NormalizeEmbedding         bool              `json:"normalize_embedding"`
	NumHiddenLayers            int               `json:"num_hidden_layers"`
	OutputPast                 bool              `json:"output_past"`
	PadTokenID                 int               `json:"pad_token_id"`
	ScaleEmbedding             bool              `json:"scale_embedding"`
	StaticPositionEmbeddings   bool              `json:"static_position_embeddings"`
	TotalFlos                  float64           `json:"total_flos"`
	VocabSize                  int               `json:"vocab_size"`
	Training                   bool              `json:"training"` // Custom for spaGO
}

func Load(file string) (Config, error) {
	var config Config
	configFile, err := os.Open(file)
	if err != nil {
		return Config{}, err
	}
	defer configFile.Close()
	err = json.NewDecoder(configFile).Decode(&config)
	if err != nil {
		return Config{}, err
	}
	return config, nil
}
