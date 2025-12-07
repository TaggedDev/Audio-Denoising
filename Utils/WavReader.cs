using NAudio.Wave;

namespace DataRelated.Utils;
public static class AudioUtils
{
    public static float[] ReadMonoWav(string filePath, out int sampleRate)
    {
        using var reader = new WaveFileReader(filePath);
        sampleRate = reader.WaveFormat.SampleRate;
        int channels = reader.WaveFormat.Channels;

        var samples = new List<float>();
        var buffer = new float[1024 * channels]; // буфер для чтения
        int read;

        var sampleProvider = reader.ToSampleProvider(); // конвертирует в ISampleProvider с float

        while ((read = sampleProvider.Read(buffer, 0, buffer.Length)) > 0)
        {
            for (int i = 0; i < read; i += channels)
            {
                float sum = 0;
                for (int ch = 0; ch < channels; ch++)
                    sum += buffer[i + ch];
                samples.Add(sum / channels);
            }
        }

        return samples.ToArray();
    }
}
