using AudioDenoise.Data;
using AudioDenoise.Models;
using AudioDenoise.Utils;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace AudioDenoise.Trainer;

// ReSharper disable once InconsistentNaming
public class DCCRNTrainer(
    DCCRNModel model,
    Device device,
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
    public void Train(IEnumerable<AudioData> dataset, string? checkpointDir, int epochs = 10, int batchSize = 8,
        double lr = 1e-3, bool shuffle = true)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        // Convert to list for easy batching (if dataset is large, consider streaming)
        List<AudioData> dataList = dataset.ToList();
        Tensor window = hann_window(winLength, dtype: ScalarType.Float32, device: CPU);
        _model.to(device);
        Adam opt = optim.Adam(_model.parameters(), lr);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Console.WriteLine($"Epoch {epoch + 1}/{epochs}");
            if (shuffle)
            {
                var rnd = new Random();
                dataList = dataList.OrderBy(_ => rnd.Next()).ToList();
            }

            int totalBatches = (int)Math.Ceiling((double)dataList.Count / batchSize);
            for (int batchIndex = 0; batchIndex < totalBatches; batchIndex++)
            {
                List<AudioData> batchItems = dataList.Skip(batchIndex * batchSize).Take(batchSize).ToList();

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

                Console.WriteLine($"Epoch {epoch + 1} Batch {batchIndex + 1}/{totalBatches} Loss: {loss.ToSingle():F6}");
            }

            // optional checkpoint
            if (string.IsNullOrEmpty(checkpointDir))
                continue;
            
            Directory.CreateDirectory(checkpointDir);
            string checkPoint = Path.Combine(checkpointDir, $"DCCRN_epoch{epoch + 1}.dat");
            // move to CPU for saving (if needed)
            _model.to(CPU);
            _model.save(checkPoint);
            Console.WriteLine($"Saved checkpoint: {checkPoint}");
            _model.to(device);
        }
    }

    /// <summary>
    /// Convert list of waveforms (float[]) to batched complex spectrogram tensor [B,2,F,T]
    /// window expected on CPU; returned tensor is moved to toDevice.
    /// </summary>
    private Tensor BatchWaveformsToSpec(List<float[]> waveforms, Tensor window, Device? toDevice = null)
    {
        Device cpu = CPU;
        Tensor[] tensors = new Tensor[waveforms.Count];
        for (int i = 0; i < waveforms.Count; i++) 
            tensors[i] = tensor(waveforms[i], dtype: ScalarType.Float32, device: cpu);
        Tensor batched = stack(tensors); // [B, N] on CPU

        Tensor spec = stft(batched, nFft, hopLength, winLength, window: window, center: true, return_complex: false);
        Tensor specPerm = spec.permute(0, 3, 1, 2);

        if (toDevice != null)
            specPerm = specPerm.to(toDevice);

        return specPerm;
    }
}