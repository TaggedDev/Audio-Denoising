using AudioDenoise.Metrics;

namespace AudioDenoise.Evaluate;

public class ModelTestResult
{
    public string ModelName { get; set; } 
    public string NoiseType { get; set; } 
    public Dictionary<IMetric, double> Metrics { get; set; } = new();

    public override string ToString()
    {
        string answer = $"{ModelName}: {NoiseType}\n\t";
        foreach (KeyValuePair<IMetric, double> metric in Metrics) 
            answer += $"{metric.Key}: {metric.Value:F4}";
        return answer;
    }
}