using MathNet.Numerics.IntegralTransforms;

namespace AudioDenoise.Eval.AudioDenoise.Eval.AudioDenoise.Eval;

public class SpectralRMSE : IMetric
{
    public string Name => "SpectralRMSE";

    /// <summary>
    /// Спектральная ошибка:
    /// 1. Вычисляем FFT чистого и обработанного сигнала
    /// 2. Берем амплитуду спектра
    /// 3. Вычисляем среднеквадратичную ошибку амплитуд
    /// </summary>
    public double Compute(float[] reference, float[] estimate, int sampleRate)
    {
        if (reference.Length != estimate.Length)
            throw new ArgumentException("Reference and estimate must have same length.");

        int n = reference.Length;

        // Конвертируем float -> complex
        var refComplex = reference.Select(v => new System.Numerics.Complex(v, 0.0)).ToArray();
        var estComplex = estimate.Select(v => new System.Numerics.Complex(v, 0.0)).ToArray();

        // FFT
        Fourier.Forward(refComplex, FourierOptions.Matlab);
        Fourier.Forward(estComplex, FourierOptions.Matlab);

        double mse = 0;
        for (int i = 0; i < n; i++)
        {
            double refMag = refComplex[i].Magnitude;
            double estMag = estComplex[i].Magnitude;

            double diff = refMag - estMag;
            mse += diff * diff;
        }

        mse /= n;
        return Math.Sqrt(mse); // RMSE
    }
}