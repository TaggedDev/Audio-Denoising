using AudioDenoise.Data;
using AudioDenoise.Metrics;
using AudioDenoise.Evaluate;
using AudioDenoise.Models;
using Microsoft.ML;

namespace AudioDenoise;

internal static class Program
{
    private static void Main()
    {
        string fileName = "test_audio.csv";

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

        List<INoiseReductionModel> models = 
        [
            new FourierNoiseReductionModel()
        ];

        List<IMetric> metrics =
        [
            new SISDRMetric(),
            new SNRMetric(),
            new SegmentalSNR(),
            new SpectralRMSE()
        ];

        // Новый evaluator, адаптированный под IDataView/IEnumerable<AudioData>
        var evaluator = new NoiseModelEvaluator(models, metrics);

        // Запуск с батчами
        List<ModelTestResult> results = evaluator
            .RunTest(audioData, batchSize: 16)
            .OrderBy(x => x.ModelName)
            .ThenBy(x => x.NoiseType)
            .ToList();

        // Печать результатов
        foreach (ModelTestResult modelTestResult in results) 
            Console.WriteLine(modelTestResult);
        
        IEnumerable<AudioData> sample = audioData.Take(5);
        foreach (INoiseReductionModel model in models) 
            model.ProcessAndWriteSample(sample);
    }
}