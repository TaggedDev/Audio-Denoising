namespace AudioDenoise.Metrics;

/// <summary>
/// Simple global Signal-to-Noise Ratio (SNR)
/// SNR = 10 * log10( sum(reference^2) / sum( (reference - estimate)^2 ) )
/// </summary>
public class SNRMetric : IMetric
{
    public string Name => "SNR";

    public double Compute(float[] reference, float[] estimate, int sampleRate)
    {
        if (reference.Length != estimate.Length) throw new ArgumentException("reference and estimate must have same length");
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < reference.Length; i++)
        {
            num += reference[i] * reference[i];
            double e = reference[i] - estimate[i];
            den += e * e;
        }
        if (den <= 0) return double.PositiveInfinity;
        return 10.0 * Math.Log10(num / den);
    }
}