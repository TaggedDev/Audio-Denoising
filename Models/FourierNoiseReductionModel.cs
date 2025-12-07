using System.Numerics;
using MathNet.Numerics.IntegralTransforms;
using MathNet.Numerics.Statistics;

namespace AudioDenoise.Models;

/// <summary>
/// Простейшая noise-reduction модель на основе FFT + пороговой маски по мощности (spectral gating).
/// Это baseline: быстро, детерминированно, пригодно для inference/оценки.
/// Для production/лучшего качества стоит заменить на STFT с оконной обработкой и адаптивной маской.
/// </summary>
public class FourierNoiseReductionModel : INoiseReductionModel
{
    /// <summary>Имя модели</summary>
    public string Name => "FourierModel";

    /// <summary>Пороговой множитель относительно медианы/средней мощности</summary>
    private readonly double _thresholdMultiplier;

    public FourierNoiseReductionModel(double thresholdMultiplier = 1.5)
    {
        // thresholdMultiplier: чем выше — тем агрессивнее подавление
        _thresholdMultiplier = thresholdMultiplier;
    }

    public void LoadModel(string modelPath)
    {
        // В простейшей реализации нет внешних весов.
        // Оставлено для совместимости с интерфейсом.
        // Если надо — можно загрузить параметры (например, thresholdMultiplier) из config.
    }

    public float[] Process(float[] noisyWaveform, int sampleRate)
    {
        if (noisyWaveform == null) throw new ArgumentNullException(nameof(noisyWaveform));
        if (noisyWaveform.Length == 0) return Array.Empty<float>();

        // 1) Сконвертировать в Complex[]
        int n = noisyWaveform.Length;
        var spectrum = new Complex[n];
        for (int i = 0; i < n; i++)
            spectrum[i] = new Complex(noisyWaveform[i], 0.0);

        // 2) Forward FFT (in-place)
        // Используем MathNet.Numerics.IntegralTransforms
        // FourierOptions.Matlab — более привычное поведение (см. доки). Можно выбирать другой.
        Fourier.Forward(spectrum, FourierOptions.Matlab);

        // 3) Оценка мощности (magnitude^2). Используем абсолютные значения спектра.
        double[] magnitudes = new double[n];
        for (int i = 0; i < n; i++)
            magnitudes[i] = spectrum[i].Magnitude * spectrum[i].Magnitude;

        // 4) Вычислим порог — медиана (устойчива к выбросам) или среднее (в качестве fallback)
        double median = magnitudes.Median();
        if (double.IsNaN(median) || median <= 0)
        {
            median = magnitudes.Average();
            if (double.IsNaN(median) || median <= 0) median = 1e-12;
        }

        double threshold = median * _thresholdMultiplier;

        // 5) Применяем маску: зануляем частоты с низкой мощностью
        for (int i = 0; i < n; i++)
        {
            if (magnitudes[i] < threshold)
            {
                spectrum[i] = Complex.Zero;
            }
            // иначе — оставляем компоненту как есть
        }

        // 6) Inverse FFT
        Fourier.Inverse(spectrum, FourierOptions.Matlab);

        // 7) Извлекаем реальную часть и нормируем при необходимости
        float[] output = new float[n];
        double maxAbs = 0.0;
        for (int i = 0; i < n; i++)
        {
            double v = spectrum[i].Real;
            output[i] = (float)v;
            double absv = Math.Abs(v);
            if (absv > maxAbs) maxAbs = absv;
        }

        // Если сигнал вышел за [-1..1], нормируем
        if (maxAbs > 1e-9 && maxAbs > 1.0)
        {
            double scale = 1.0 / maxAbs;
            for (int i = 0; i < n; i++)
                output[i] = (float)(output[i] * scale);
        }

        return output;
    }
}