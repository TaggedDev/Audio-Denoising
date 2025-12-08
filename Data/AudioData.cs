using Microsoft.ML.Data;

namespace AudioDenoise.Data;

public class AudioData
{
    [LoadColumn(0)]
    public string Id { get; set; } = string.Empty;

    [LoadColumn(1)]
    public string FileName { get; set; } = string.Empty;

    [LoadColumn(2)]
    public string CleanPath { get; set; } = string.Empty;

    [LoadColumn(3)]
    public string NoisyPath { get; set; } = string.Empty;

    [LoadColumn(4)]
    public string NoiseType { get; set; } = string.Empty;
}