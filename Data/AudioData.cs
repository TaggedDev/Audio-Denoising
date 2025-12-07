using Microsoft.ML.Data;

namespace AudioDenoise.Data;

public class AudioData
{
    [LoadColumn(0)]
    public string Id { get; set; } = string.Empty;

    [LoadColumn(1)]
    public string FileName { get; set; } = string.Empty;

    // Если вы заранее знаете порядок колонок в CSV, добавьте [LoadColumn(index)].
    [LoadColumn(2)]
    public string CleanPath { get; set; } = string.Empty;

    [LoadColumn(3)]
    public string NoisyPath { get; set; } = string.Empty;

    [LoadColumn(4)]
    public float CleanDuration { get; set; } = 0f;

    [LoadColumn(5)]
    public float NoisyDuration { get; set; } = 0f;

    [LoadColumn(6)]
    public string Kind { get; set; } = string.Empty; // "Train" / "Test"
}