using AudioDenoise.Data;
using AudioDenoise.Metrics;
using AudioDenoise.Models;
using AudioDenoise.Trainer;
using AudioDenoise.Utils;
using ShellProgressBar;
using TorchSharp;
using static TorchSharp.torch;

namespace AudioDenoise.Evaluate;

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
    public List<ModelTestResult> RunTest(List<AudioData> noisyData, int batchSize = 8)
    {
        List<ModelTestResult> results = [];

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

                    results.Add(new ModelTestResult
                    {
                        ModelName = model.Name,
                        NoiseType = audioData.NoiseType,
                        Metrics = _metrics.ToDictionary(
                            x => x,
                            x => x.Compute(reference, denoised, noisySampleRate))
                    });
                }

                pbar.Tick($"Processed {audioData.FileName} [{audioData.NoiseType}]");
            }
        }

        List<ModelTestResult> aggregatedResults = [];

        var modelNoiseGroups = results
            .GroupBy(r => new { r.ModelName, r.NoiseType })
            .Select(g => new ModelTestResult()
            {
                ModelName = g.Key.ModelName,
                NoiseType = g.Key.NoiseType,
                Metrics = g.SelectMany(r => r.Metrics)
                    .GroupBy(m => m.Key)
                    .ToDictionary(
                        group => group.Key,
                        group => group.Average(m => m.Value)
                    )
            });

        aggregatedResults.AddRange(modelNoiseGroups);

        // 2. Для каждой модели по всем типам шума (общий результат по модели)
        var modelGroups = results
            .GroupBy(r => r.ModelName)
            .Select(g => new ModelTestResult
            {
                ModelName = g.Key,
                NoiseType = "All",
                Metrics = g.SelectMany(r => r.Metrics)
                    .GroupBy(m => m.Key)
                    .ToDictionary(
                        group => group.Key,
                        group => group.Average(m => m.Value)
                    )
            });

        aggregatedResults.AddRange(modelGroups);
        return aggregatedResults;
    }
}