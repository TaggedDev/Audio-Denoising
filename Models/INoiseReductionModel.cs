namespace AudioDenoise.Models;

/// <summary>
/// Интерфейс для noise-reduction моделей.
/// </summary>
public interface INoiseReductionModel
{
    /// <summary>Человеко-читаемое имя модели.</summary>
    string Name { get; }

    /// <summary>
    /// Выполнить inference: на входе моно сигнал (float[]), нормированный в [-1..1], и sampleRate.
    /// Возвращает оценённый (денойзенный) сигнал того же размера (или обрезанный/паддированный до исходной длины).
    /// </summary>
    float[] Process(float[] noisyWaveform, int sampleRate);

    /// <summary>
    /// (Опционально) загрузить веса/гиперпараметры модели.
    /// </summary>
    void LoadModel(string modelPath);
}