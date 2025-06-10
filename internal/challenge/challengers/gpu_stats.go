package challengers

import (
	"fmt"
	"os/exec"
	"strings"

	"go.uber.org/zap"
)

// GPUStatsChallenger polls GPU stats.
type GPUStatsChallenger struct{}

// Execute polls the metadata of the GPUs using nvidia-smi.
func (c *GPUStatsChallenger) Execute(payload interface{}, log *zap.Logger) (interface{}, error) {
	log.Info("Polling GPU stats...")
	cmd := exec.Command("nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			log.Error("nvidia-smi failed", zap.Error(exitErr), zap.String("stderr", string(exitErr.Stderr)))
			// nvidia-smi not found, handle this case gracefully
			if strings.Contains(string(exitErr.Stderr), "executable file not found") || strings.Contains(string(exitErr.Stderr), "No such file or directory") {
				log.Warn("nvidia-smi command not found, skipping GPU stats poll")
				return map[string]interface{}{"error": "nvidia-smi not found"}, nil
			}
		}
		log.Error("Failed to execute nvidia-smi", zap.Error(err))
		return nil, err
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	stats := make(map[string]interface{})
	for i, line := range lines {
		values := strings.Split(line, ", ")
		gpuStats := map[string]interface{}{
			"name":                values[0],
			"driver_version":      values[1],
			"memory_total_mb":     values[2],
			"memory_used_mb":      values[3],
			"memory_free_mb":      values[4],
			"utilization_gpu_pct": values[5],
			"utilization_mem_pct": values[6],
		}
		stats[fmt.Sprintf("gpu_%d", i)] = gpuStats
	}

	log.Info("Successfully polled GPU stats")
	return stats, nil
}
