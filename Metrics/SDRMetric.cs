namespace AudioDenoise.Eval;

/// <summary>
/// Classic (windowed) SDR as in BSS Eval is more involved; here we provide a simple global SDR
/// SDR = 10 * log10( sum(reference^2) / sum( (reference - estimate)^2 ) )
/// (same as SNR above) — keep for API compatibility.
/// </summary>
public class SDRMetric : IMetric
{
    public string Name => "SDR";
    public double Compute(float[] reference, float[] estimate, int sampleRate)
    {
        // reuse SNR formula
        var snr = new SNRMetric();
        return snr.Compute(reference, estimate, sampleRate);
    }
}