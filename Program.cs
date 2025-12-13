using AudioDenoise.Data;
using AudioDenoise.Metrics;
using AudioDenoise.Evaluate;
using AudioDenoise.Models;
using AudioDenoise.Trainer;
using AudioDenoise.Utils;
using Microsoft.ML;
using TorchSharp;

namespace AudioDenoise;

internal static class Program
{
    private static void Main()
    {
        const string trainAudio = "train_audio.csv";
        const string testAudio = "train_audio.csv";
        const string checkpointDir = "checkpoints_test";
        
        MLContext mlContext = new MLContext();
        IDataView trainDataView = mlContext.Data.LoadFromTextFile<AudioData>(trainAudio, separatorChar: ',', hasHeader: true);
        List<AudioData> trainData = mlContext.Data.CreateEnumerable<AudioData>(trainDataView, reuseRowObject: false).ToList();

        IDataView testDataView = mlContext.Data.LoadFromTextFile<AudioData>(testAudio, separatorChar: ',', hasHeader: true);
        List<AudioData> testData = mlContext.Data.CreateEnumerable<AudioData>(testDataView, reuseRowObject: false).ToList();
        
        torch.Device device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
        DCCRNModel model = new DCCRNModel(device);
        List<IMetric> metrics =
        [
            new SISDRMetric(),
            new SNRMetric(),
            new SegmentalSNR(),
            new SpectralRMSE()
        ];
        DCCRNTrainer trainer = new DCCRNTrainer(model, device, metrics, nFft: 512, winLength: 512, hopLength: 128);
        
        Directory.CreateDirectory(checkpointDir);
        trainer.Run(trainData, testData, checkpointDir, epochs: 5, batchSize: 12, lr: 1e-3);
    }

    private static void TestModelLoading(string checkpointDir, torch.Device device, List<AudioData> data)
    {
        string checkpointPath = Path.Combine(checkpointDir, "DCCRN_epoch1.dat");
        if (!File.Exists(checkpointPath))
        {
            Console.WriteLine("Checkpoint not found!");
            return;
        }
        Console.WriteLine($"Checkpoint saved at: {checkpointPath}");

        // Load into fresh model and run inference on first sample
        var loadedModel = new DCCRNModel(device);
        loadedModel.to(torch.CPU); // ensure load on CPU
        loadedModel.load(checkpointPath);
        loadedModel.to(device);
        var sample = data.First();
        float[] noisy = AudioUtils.ReadMonoWav(sample.NoisyPath, out int sr);
        var denoised = loadedModel.Process(noisy, sr);

        Console.WriteLine($"Denoised sample length: {denoised.Length}");
    }

    private static void Inference()
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
            new FourierNoiseReductionModel(),
            new DCCRNModel(torch.cuda.is_available() ? torch.CUDA : torch.CPU)
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