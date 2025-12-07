namespace AudioDenoise.Eval;

/// <summary>
/// Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
/// Following definition: compute optimal scaling alpha = (s^T s_hat) / (s^T s)
/// then s_target = alpha * s; e = s_hat - s_target; SI-SDR = 10 log10( ||s_target||^2 / ||e||^2 ).
/// </summary>
public class SISDRMetric : IMetric
{
    public string Name => "SI-SDR";

    public double Compute(float[] reference, float[] estimate, int sampleRate)
    {
        if (reference.Length != estimate.Length) throw new ArgumentException("reference and estimate must have same length");

        // Convert to double for accumulation
        double dot = 0.0;
        double refEnergy = 0.0;
        for (int i = 0; i < reference.Length; i++)
        {
            dot += reference[i] * estimate[i];
            refEnergy += reference[i] * reference[i];
        }
        if (refEnergy == 0.0) return double.NegativeInfinity;
        double alpha = dot / refEnergy;

        double targetEnergy = 0.0;
        double errEnergy = 0.0;
        for (int i = 0; i < reference.Length; i++)
        {
            double sTarget = alpha * reference[i];
            targetEnergy += sTarget * sTarget;
            double err = estimate[i] - sTarget;
            errEnergy += err * err;
        }
        if (errEnergy == 0.0) return double.PositiveInfinity;
        return 10.0 * Math.Log10(targetEnergy / errEnergy);
    }
}
