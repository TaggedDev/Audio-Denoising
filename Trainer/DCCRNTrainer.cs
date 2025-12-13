using AudioDenoise.Data;
using AudioDenoise.Metrics;
using AudioDenoise.Models;
using AudioDenoise.Utils;
using ShellProgressBar;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace AudioDenoise.Trainer;

// ReSharper disable once InconsistentNaming
public class DCCRNTrainer(
    DCCRNModel model,
    Device device,
    List<IMetric> metrics,
    int nFft = 512,
    int winLength = 512,
    int hopLength = 128)
{
    private readonly DCCRNModel _model = model ?? throw new ArgumentNullException(nameof(model));

    /// <summary>
    /// Train on an enumerable of AudioData (assumed to contain paths to noisy/clean WAVs).
    /// lossType: MSE
    /// Returns trained model (same instance).
    /// </summary>
    public void Run(IEnumerable<AudioData> trainDataset, IEnumerable<AudioData> testDataset, string? checkpointDir,
        int epochs = 10, int batchSize = 32, double lr = 1e-3, bool trainShuffle = true, int saveEveryEpoch = 5)
    {
        ArgumentNullException.ThrowIfNull(trainDataset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        // Convert to list for easy batching (if dataset is large, consider streaming)
        List<AudioData> dataList = trainDataset.ToList();
        Tensor window = hann_window(winLength, dtype: ScalarType.Float32, device: CPU);
        _model.to(device);
        Adam opt = optim.Adam(_model.parameters(), lr);

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            TrainModel(batchSize, trainShuffle, dataList, window, opt, epoch);
            //TestWithMetrics(testDataset, batchSize);

            // optional checkpoint
            if (string.IsNullOrEmpty(checkpointDir) || epoch % saveEveryEpoch == 0)
                continue;
            
            Directory.CreateDirectory(checkpointDir);
            string checkPoint = Path.Combine(checkpointDir, $"DCCRN_epoch{epoch}.dat");
            _model.save(checkPoint);
            Console.WriteLine($"Saved checkpoint: {checkPoint}");
        }
    }

    private void TrainModel(int batchSize, bool trainShuffle, List<AudioData> trainData, Tensor window, Adam opt,
        int epoch)
    {
        if (trainShuffle)
        {
            var rnd = new Random();
            trainData = trainData.OrderBy(_ => rnd.Next()).ToList();
        }
        
        var progressOptions = new ProgressBarOptions
        {
            ProgressCharacter = '─',
            ProgressBarOnBottom = true,
            ForegroundColor = ConsoleColor.DarkRed,
            BackgroundColor = ConsoleColor.DarkGray,
            DisplayTimeInRealTime = true,
            CollapseWhenFinished = false
        };
        
        int totalBatches = (int)Math.Ceiling((double)trainData.Count / batchSize);
        using var pbar = new ProgressBar(totalBatches, "Testing model…", progressOptions);
        
        for (int batchIndex = 0; batchIndex < totalBatches; batchIndex++)
        {
            List<AudioData> batchItems = trainData.Skip(batchIndex * batchSize).Take(batchSize).ToList();

            // Read waveforms and sampleRates
            List<float[]> waveformsNoisy = [];
            List<float[]> waveformsClean = [];

            foreach (var ad in batchItems)
            {
                float[] noisy = AudioUtils.ReadMonoWav(ad.NoisyPath, out _);
                float[] clean = AudioUtils.ReadMonoWav(ad.CleanPath, out _);

                waveformsNoisy.Add(noisy);
                waveformsClean.Add(clean);
            }

            int maxLen = waveformsNoisy.Concat(waveformsClean).Max(x => x.Length);
            for (int i = 0; i < waveformsNoisy.Count; i++)
            {
                float[] padded = new float[maxLen];
                if (waveformsNoisy[i].Length < maxLen)
                {
                    Array.Copy(waveformsNoisy[i], padded, waveformsNoisy[i].Length);
                    waveformsNoisy[i] = padded;
                }

                if (waveformsClean[i].Length >= maxLen)
                    continue;

                Array.Copy(waveformsClean[i], padded, waveformsClean[i].Length);
                waveformsClean[i] = padded;
            }


            // Build batch spectrograms [B,2,F,T]
            Tensor noisySpecB = BatchWaveformsToSpec(waveformsNoisy, window, toDevice: device);
            Tensor cleanSpecB = BatchWaveformsToSpec(waveformsClean, window, toDevice: device);

            // Train step
            _model.train();
            opt.zero_grad();

            // forward
            Tensor estSpecB = _model.forward(noisySpecB); // expects [B,2,F,T]
            Tensor loss = nn.functional.mse_loss(estSpecB, cleanSpecB);
            loss.backward();
            opt.step();

            pbar.Tick($"Epoch {epoch} Batch {batchIndex + 1}/{totalBatches} Loss: {loss.ToSingle():F6}");
        }
    }

    /// <summary>
    /// Evaluate the model on a dataset with batching (no shuffling) and print batch-wise metrics.
    /// Uses a list of IMetric objects to compute aggregated scores per batch.
    /// </summary>
    private void TestWithMetrics(IEnumerable<AudioData> testSet, int batchSize)
    {
        ArgumentNullException.ThrowIfNull(testSet);
        ArgumentNullException.ThrowIfNull(metrics);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        List<AudioData> testList = testSet.ToList();
        Tensor window = hann_window(winLength, dtype: ScalarType.Float32, device: CPU);

        _model.eval(); // evaluation mode
        
        var progressOptions = new ProgressBarOptions
        {
            ProgressCharacter = '─',
            ProgressBarOnBottom = true,
            ForegroundColor = ConsoleColor.Cyan,
            BackgroundColor = ConsoleColor.DarkGray,
            DisplayTimeInRealTime = true,
            CollapseWhenFinished = false
        };
        
        int total = (int)Math.Ceiling((double)testList.Count / batchSize);
        using var pbar = new ProgressBar(total, "Testing model…", progressOptions);

        for (int batchIndex = 0; batchIndex < total; batchIndex++)
        {
            List<AudioData> batchItems = testList.Skip(batchIndex * batchSize).Take(batchSize).ToList();

            // Read waveforms
            List<float[]> waveformsNoisy = [];
            List<float[]> waveformsClean = [];
            foreach (var ad in batchItems)
            {
                waveformsNoisy.Add(AudioUtils.ReadMonoWav(ad.NoisyPath, out _));
                waveformsClean.Add(AudioUtils.ReadMonoWav(ad.CleanPath, out _));
            }

            int maxLen = waveformsNoisy.Concat(waveformsClean).Max(x => x.Length);
            for (int i = 0; i < waveformsNoisy.Count; i++)
            {
                float[] padded = new float[maxLen];
                Array.Copy(waveformsNoisy[i], padded, waveformsNoisy[i].Length);
                waveformsNoisy[i] = padded;

                Array.Copy(waveformsClean[i], padded, waveformsClean[i].Length);
                waveformsClean[i] = padded;
            }

            // Spectrograms
            Tensor noisySpecB = BatchWaveformsToSpec(waveformsNoisy, window, toDevice: device);
            Tensor cleanSpecB = BatchWaveformsToSpec(waveformsClean, window, toDevice: device);

            // Forward pass
            using var estSpecB = _model.forward(noisySpecB); // [B,2,F,T]
            using var lossTensor = nn.functional.mse_loss(estSpecB, cleanSpecB);
            float mseLoss = lossTensor.ToSingle();

            // Convert to waveform
            List<float[]> denoisedWaveforms = new List<float[]>(batchItems.Count);
            for (int i = 0; i < batchItems.Count; i++)
            {
                using var singleEstSpec = estSpecB[i].unsqueeze(0); // [1,2,F,T]
                float[] denoised = _model.SpectrogramToWaveform(singleEstSpec, waveformsClean[i].Length);
                denoisedWaveforms.Add(denoised);
            }

            // Compute metrics per batch
            string metricsStr = string.Join(" | ", metrics.Select(metric =>
            {
                // Compute metric for each sample in batch
                IEnumerable<double> values = batchItems.Select((_, idx) =>
                {
                    const int sampleRate = 32000;
                    return metric.Compute(waveformsClean[idx], denoisedWaveforms[idx], sampleRate);
                });
                double avg = values.Average();
                return $"{metric.Name}: {avg:F4}";
            }));
            
            pbar.Tick($"Test Batch {batchIndex + 1}/{total} | Loss: {mseLoss:F6} | {metricsStr}");
        }
    }

    /// <summary>
    /// Convert list of waveforms (float[]) to batched complex spectrogram tensor [B,2,F,T]
    /// window expected on CPU; returned tensor is moved to toDevice.
    /// </summary>
    private Tensor BatchWaveformsToSpec(List<float[]> waveforms, Tensor window, Device toDevice)
    {
        Tensor[] tensors = new Tensor[waveforms.Count];
        for (int i = 0; i < waveforms.Count; i++) 
            tensors[i] = tensor(waveforms[i], dtype: ScalarType.Float32, device: toDevice);
        Tensor batched = stack(tensors); // [B, N] on CPU

        window = window.to(toDevice);
        Tensor spec = stft(batched, nFft, hopLength, winLength, window: window, center: true, 
            return_complex: false);
        return spec.permute(0, 3, 1, 2).to(toDevice);
    }
}