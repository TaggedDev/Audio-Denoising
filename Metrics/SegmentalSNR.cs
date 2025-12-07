using AudioDenoise.Eval;

public class SegmentalSNR : IMetric
{
    public string Name => "SegmentalSNR";

    /// <summary>
    /// Segmental SNR. Разбиваем сигнал на фреймы, вычисляем SNR на каждом фрейме и усредняем.
    /// Формула для фрейма i:
    /// SNR_i = 10 * log10( sum(ref^2) / sum((ref - est)^2) )
    /// </summary>
    public double Compute(float[] reference, float[] estimate, int sampleRate)
    {
        if (reference.Length != estimate.Length)
            throw new ArgumentException("Reference and estimate must have same length.");

        int frameSize = sampleRate / 50; // 20 ms
        int hop = frameSize; // без перекрытия
        int totalFrames = (int)Math.Ceiling(reference.Length / (double)hop);

        double snrSum = 0;
        int count = 0;

        for (int i = 0; i < totalFrames; i++)
        {
            int start = i * hop;
            int len = Math.Min(frameSize, reference.Length - start);

            double signalEnergy = 0;
            double noiseEnergy = 0;

            for (int j = 0; j < len; j++)
            {
                double refVal = reference[start + j];
                double estVal = estimate[start + j];

                signalEnergy += refVal * refVal;
                noiseEnergy += (refVal - estVal) * (refVal - estVal);
            }

            if (noiseEnergy > 0)
            {
                double snrFrame = 10 * Math.Log10(signalEnergy / noiseEnergy);
                snrSum += snrFrame;
                count++;
            }
        }

        return count > 0 ? snrSum / count : double.NaN;
    }
}