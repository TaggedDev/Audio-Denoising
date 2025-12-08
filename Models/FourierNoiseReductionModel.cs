using System.Numerics;
using AudioDenoise.Data;
using AudioDenoise.Utils;
using MathNet.Numerics.IntegralTransforms;
using MathNet.Numerics.Statistics;

namespace AudioDenoise.Models;

/// <summary>
/// Простейшая noise-reduction модель на основе FFT + пороговой маски по мощности (spectral gating).
/// Это baseline: быстро, детерминированно, пригодно для inference/оценки.
/// Для production/лучшего качества стоит заменить на STFT с оконной обработкой и адаптивной маской.
/// </summary>
public class FourierNoiseReductionModel(double thresholdMultiplier = 1.5) : INoiseReductionModel
{
    /// <summary>Имя модели</summary>
    public string Name => "FourierModel";

    /// <summary>Пороговой множитель относительно медианы мощности</summary>
    private readonly double _thresholdMultiplier = thresholdMultiplier;

    // thresholdMultiplier: чем выше — тем агрессивнее подавление

    public void LoadModel(string modelPath)
    {
        // В простейшей реализации нет внешних весов.
        // Оставлено для совместимости с интерфейсом.
        // Если надо — можно загрузить параметры (например, thresholdMultiplier) из config.
    }

    public void ProcessAndWriteSample(IEnumerable<AudioData> sample)
    {
        // 1) Папка для вывода
        string baseOutputDir = Path.Combine("outputs", Name);
        Directory.CreateDirectory(baseOutputDir);

        foreach (AudioData audioData in sample)
        {
            // Получаем базовое имя файла без расширения
            string baseFileName = Path.GetFileNameWithoutExtension(audioData.FileName);
            string extension = Path.GetExtension(audioData.FileName);

            // Читаем все сигналы
            float[] noisySignal = AudioUtils.ReadMonoWav(audioData.NoisyPath, out int sampleRate);
            float[] processedSignal = Process(noisySignal, sampleRate);
            float[] cleanSignal = AudioUtils.ReadMonoWav(audioData.CleanPath, out int cleanSampleRate);

            // Нормализуем processed сигнал если нужно
            float maxVal = processedSignal.Max(Math.Abs);
            if (maxVal > 1.0f)
            {
                for (int i = 0; i < processedSignal.Length; i++)
                    processedSignal[i] /= maxVal;
            }

            // Сохраняем noisy (input) файл
            AudioUtils.SaveAudioFile(
                Path.Combine(baseOutputDir, $"{baseFileName} (input){extension}"),
                noisySignal,
                sampleRate
            );

            // Сохраняем processed (output) файл
            AudioUtils.SaveAudioFile(
                Path.Combine(baseOutputDir, $"{baseFileName} (output){extension}"),
                processedSignal,
                sampleRate
            );

            // Сохраняем clean (truth) файл
            AudioUtils.SaveAudioFile(
                Path.Combine(baseOutputDir, $"{baseFileName} (truth){extension}"),
                cleanSignal,
                cleanSampleRate
            );
        }
    }

    public float[] Process(float[] noisyWaveform, int sampleRate)
    {
        ArgumentNullException.ThrowIfNull(noisyWaveform);
        if (noisyWaveform.Length == 0) 
            return [];

        int n = noisyWaveform.Length;
        var spectrum = new Complex[n];
        for (int i = 0; i < n; i++)
            spectrum[i] = new Complex(noisyWaveform[i], 0.0);

        // 2) Forward FFT (in-place)
        Fourier.Forward(spectrum, FourierOptions.Matlab);

        // 3) Оценка мощности (magnitude^2)
        double[] magnitudes = new double[n];
        for (int i = 0; i < n; i++)
            magnitudes[i] = spectrum[i].Magnitude * spectrum[i].Magnitude;

        // 4) Вычислим порог — медиана (устойчива к выбросам) или среднее (в качестве fallback)
        double median = magnitudes.Median();
        double threshold = median * _thresholdMultiplier;

        // 5) Применяем маску: зануляем частоты с низкой мощностью
        for (int i = 0; i < n; i++)
            if (magnitudes[i] < threshold)
                spectrum[i] = Complex.Zero;

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
        if (!(maxAbs > 1e-9) || !(maxAbs > 1.0))
            return output;

        double scale = 1.0 / maxAbs;
        for (int i = 0; i < n; i++)
            output[i] = (float)(output[i] * scale);

        return output;
    }
}