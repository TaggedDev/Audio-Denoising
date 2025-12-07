using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace AudioDenoise.Utils;
public static class AudioUtils
{
public static float[] ReadMonoWav(string rawPath, out int sampleRate)
    {
        if (string.IsNullOrWhiteSpace(rawPath))
            throw new ArgumentException("Пустой путь к аудиофайлу.", nameof(rawPath));

        // 1) Убираем кавычки и лишние пробелы
        var p = rawPath.Trim().Trim('"', '\'');

        // 2) Если путь не корневой, приводим к абсолютному относительно базовой директории приложения
        if (!Path.IsPathRooted(p))
        {
            // Можно заменить AppContext.BaseDirectory на вашу корневую папку, если нужно.
            p = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, p));
        }
        else
        {
            // Получаем корректную форму абс. пути (удалит лишние .\ и ..\)
            p = Path.GetFullPath(p);
        }

        if (!File.Exists(p))
        {
            // Логируем точный путь который мы попытались открыть
            throw new FileNotFoundException($"Аудиофайл не найден по пути: {p}", p);
        }

        // 3) Чтение с помощью AudioFileReader (удобно: возвращает float samples через ISampleProvider)
        using var afr = new AudioFileReader(p); // поддерживает WAV/MP3 и др.
        sampleRate = afr.WaveFormat.SampleRate;

        ISampleProvider provider = afr;

        // Если у нас больше одного канала — преобразуем в моно (усреднение левого и правого)
        if (afr.WaveFormat.Channels == 2)
        {
            // StereoToMonoSampleProvider есть в NAudio.Wave.SampleProviders
            var stereoToMono = new StereoToMonoSampleProvider(afr)
            {
                LeftVolume = 0.5f,
                RightVolume = 0.5f
            };
            provider = stereoToMono;
        }
        else if (afr.WaveFormat.Channels > 2)
        {
            // Для >2 каналов можно усреднить все каналы вручную через ToSampleProvider и чтение в буфер
            provider = afr.ToSampleProvider();
        }

        // Читаем все сэмплы в памяти (если очень большие файлы — можно читать блоками)
        var samples = new List<float>();
        var buffer = new float[sampleRate * afr.WaveFormat.Channels]; // размер буфера: 1 секунда
        int read;
        while ((read = provider.Read(buffer, 0, buffer.Length)) > 0)
        {
            // если у нас >1 канал после provider (маловероятно, т.к. stereo->mono сделал 1 канал),
            // то buffer содержит моно-значения только если provider — моно; иначе нужно интерпретировать.
            for (int i = 0; i < read; i++)
                samples.Add(buffer[i]);
        }

        return samples.ToArray();
    }
}
