package openai

import (
	"bytes"
	"io"
	"net/http"

	"github.com/fxnlabs/function-node/internal/config"
	"go.uber.org/zap"
)

// NewOAIHandler creates a new http.HandlerFunc that proxies requests to the given backendURL.
func NewOAIHandler(backendConfig *config.ModelBackendConfig, log *zap.Logger) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		backendURL := backendConfig.GetModelBackendURL(r, log)
		if backendURL == "" {
			http.Error(w, "model not found or could not parse model from request", http.StatusBadRequest)
			return
		}
		NewProxyHandler(backendURL)(w, r)
	}
}

// NewProxyHandler creates a new http.HandlerFunc that proxies requests to the given backendURL.
func NewProxyHandler(backendURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		resp, err := proxyRequest(r, backendURL)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer resp.Body.Close()

		copyHeaders(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
	}
}

func proxyRequest(r *http.Request, backendURL string) (*http.Response, error) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	r.Body.Close()

	// Create a new request to the backend URL
	proxyReq, err := http.NewRequest(r.Method, backendURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	// Copy headers from the original request to the new one
	proxyReq.Header = make(http.Header)
	copyHeaders(proxyReq.Header, r.Header)

	// Send the request
	client := &http.Client{}
	resp, err := client.Do(proxyReq)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func copyHeaders(dst, src http.Header) {
	for k, vv := range src {
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}
