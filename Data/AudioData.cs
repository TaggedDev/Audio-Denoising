namespace DataRelated.Data;
using Microsoft.ML.Data;

public class AudioData
{
    [LoadColumn(0)]
    public string ID { get; set; }

    [LoadColumn(1)]
    public string Filename { get; set; }

    [LoadColumn(2)]
    public string Set { get; set; }

    [LoadColumn(3)]
    public string Kind { get; set; }

    [LoadColumn(4)]
    public string NoisyFilePath { get; set; }

    [LoadColumn(5)]
    public string CleanFilePath { get; set; }

    [LoadColumn(6)]
    public string NoiseType { get; set; }

    [LoadColumn(7)]
    public float DurationSeconds { get; set; }
}
