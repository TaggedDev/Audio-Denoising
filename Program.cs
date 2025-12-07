using AudioDenoise.Eval;
using DataRelated.Data;
using Microsoft.ML;

namespace DataRelated;

internal static class Program
{
    private static void Main()
    {
        string fileName = "train_audio.csv";

        MLContext mlContext = new MLContext();

        // Загружаем CSV через ML.NET
        IDataView dataView = mlContext.Data.LoadFromTextFile<AudioData>(
            path: fileName,
            separatorChar: ',',
            hasHeader: true
        );

        // Превращаем IDataView в IEnumerable<AudioData>
        List<AudioData> audioData =
            mlContext.Data.CreateEnumerable<AudioData>(dataView, reuseRowObject: false).ToList();

        // Выбираем только Noisy файлы для обработки
        List<AudioData> noisyData = audioData
            .Where(a => a.Set == "noisy")
            .ToList();

        List<INoiseReductionModel> models = 
        [
            new FourierNoiseReductionModel()
        ];

        List<IMetric> metrics =
        [
            new SISDRMetric(),
            new SNRMetric()
        ];

        // Новый evaluator, адаптированный под IDataView/IEnumerable<AudioData>
        var evaluator = new NoiseModelEvaluator(models, metrics);

        // Запуск с батчами
        Dictionary<string, Dictionary<string, double>> results =
            evaluator.RunTest(noisyData, audioData, batchSize: 16);

        // Печать результатов
        foreach (var modelName in results.Keys)
        {
            Console.WriteLine($"Model: {modelName}");
            foreach (var metricName in results[modelName].Keys)
            {
                Console.WriteLine($"{metricName}: {results[modelName][metricName]:F2}");
            }
        }
    }
}