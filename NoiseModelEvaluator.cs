using AudioDenoise.Data;
using AudioDenoise.Eval;
using AudioDenoise.Models;
using AudioDenoise.Utils;
using ShellProgressBar;

namespace AudioDenoise;

public class NoiseModelEvaluator(IEnumerable<INoiseReductionModel> models, IEnumerable<IMetric> metrics)
{
    private readonly List<INoiseReductionModel> _models =
        models.ToList() ?? throw new ArgumentNullException(nameof(models));

    private readonly List<IMetric> _metrics = metrics.ToList() ?? throw new ArgumentNullException(nameof(metrics));

    /// <summary>
    /// Run evaluation.
    /// noisyData: list of noisy AudioData rows
    /// allData: full dataset to get references (Clean files)
    /// </summary>
    public Dictionary<string, Dictionary<string, double>> RunTest(
        List<AudioData> noisyData,
        int batchSize = 8)
    {
        var results = new Dictionary<string, Dictionary<string, List<double>>>();
        foreach (var model in _models)
            results[model.Name] = _metrics.ToDictionary(m => m.Name, _ => new List<double>());

        int total = noisyData.Count;

        var progressOptions = new ProgressBarOptions
        {
            ProgressCharacter = '─',
            ProgressBarOnBottom = true,
            ForegroundColor = ConsoleColor.Cyan,
            BackgroundColor = ConsoleColor.DarkGray,
            DisplayTimeInRealTime = true,
            CollapseWhenFinished = false
        };

        using var pbar = new ProgressBar(total, "Processing audio files…", progressOptions);

        for (int start = 0; start < total; start += batchSize)
        {
            int end = Math.Min(start + batchSize, total);
            IEnumerable<AudioData> currentBatch = noisyData.Skip(start).Take(end - start);

            foreach (var audioData in currentBatch)
            {
                // читаем аудио
                float[] noisySignal = AudioUtils.ReadMonoWav(audioData.NoisyPath, out int noisySampleRate);

                var modelOutputs = new Dictionary<string, float[]>();
                foreach (INoiseReductionModel model in _models)
                    modelOutputs[model.Name] = model.Process(noisySignal, noisySampleRate);

                float[] cleanSignal = AudioUtils.ReadMonoWav(audioData.CleanPath, out _);

                int length = Math.Min(cleanSignal.Length, noisySignal.Length);
                float[] reference = cleanSignal.Take(length).ToArray();

                foreach (var model in _models)
                {
                    float[] denoised = modelOutputs[model.Name].Take(length).ToArray();

                    foreach (var metric in _metrics)
                    {
                        double val = metric.Compute(reference, denoised, noisySampleRate);
                        results[model.Name][metric.Name].Add(val);
                    }
                }

                pbar.Tick($"Processed {audioData.FileName}");
            }
        }

        // усреднение
        var meanResults = new Dictionary<string, Dictionary<string, double>>();
        foreach (var model in _models)
        {
            meanResults[model.Name] = new Dictionary<string, double>();
            foreach (var metric in _metrics)
            {
                List<double> vals = results[model.Name][metric.Name];
                meanResults[model.Name][metric.Name] =
                    vals.Count == 0 ? double.NaN : vals.Average();
            }
        }

        return meanResults;
    }
}