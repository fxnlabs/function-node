package openai

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/fxnlabs/function-node/internal/config"
	"go.uber.org/zap"
)

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
	FxnID   string `json:"fxn_id"`
}

type ModelList struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// NewOAIProxyHandler creates a new http.HandlerFunc that proxies requests to the given backendURL.
func NewOAIProxyHandler(cfg *config.Config, modelBackendConfig *config.ModelBackendConfig, log *zap.Logger) http.HandlerFunc {
	tr := &http.Transport{
		MaxIdleConns:    cfg.Proxy.MaxIdleConns,
		IdleConnTimeout: cfg.Proxy.IdleConnTimeout,
	}
	client := &http.Client{
		Transport: tr,
	}
	return func(w http.ResponseWriter, r *http.Request) {
		modelBackend, err := modelBackendConfig.GetModelBackend(r, log)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		proxyRequest(r, modelBackend, w, client)
	}
}

func NewModelsHandler(backendConfig *config.ModelBackendConfig, log *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		models := make([]Model, 0, len(backendConfig.Models))
		for modelID, modelInfo := range backendConfig.Models {
			models = append(models, Model{
				ID:      modelID,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "fxn",
				FxnID:   modelInfo.FxnID,
			})
		}

		modelList := ModelList{
			Object: "list",
			Data:   models,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(modelList); err != nil {
			http.Error(w, "Failed to encode models", http.StatusInternalServerError)
		}
	}
}

func proxyRequest(r *http.Request, modelBackend *config.ModelBackend, w http.ResponseWriter, client *http.Client) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	r.Body.Close()

	// Create a new request to the backend URL
	target, err := url.Parse(modelBackend.URL)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	target = target.JoinPath(r.URL.Path)

	proxyReq, err := http.NewRequest(r.Method, target.String(), bytes.NewReader(body))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	// Copy headers from the original request to the new one
	proxyReq.Header = make(http.Header)
	copyHeaders(proxyReq.Header, r.Header)

	if modelBackend.Auth != nil {
		if modelBackend.Auth.APIKey != "" {
			proxyReq.Header.Set("x-api-key", modelBackend.Auth.APIKey)
		} else if modelBackend.Auth.BearerToken != "" {
			proxyReq.Header.Set("Authorization", "Bearer "+modelBackend.Auth.BearerToken)
		}
	}

	// Send the request
	resp, err := client.Do(proxyReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func copyHeaders(dst, src http.Header) {
	for k, vv := range src {
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}
