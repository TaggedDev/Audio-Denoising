namespace AudioDenoise.Metrics;

public interface IMetric
{
    /// <summary>
    /// Metric name (e.g. "SNR", "SI-SDR", "STOI")
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Compute metric value given clean (reference) and estimated (denoised) signals.
    /// Signals must have same length and be single-channel PCM normalized to [-1..1].
    /// </summary>
    double Compute(float[] reference, float[] estimate, int sampleRate);
}