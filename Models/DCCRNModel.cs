using AudioDenoise.Data;
using AudioDenoise.Utils;
using NAudio.Wave;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace AudioDenoise.Models;

// ReSharper disable once InconsistentNaming
public class DCCRNModel : nn.Module<Tensor, Tensor>, INoiseReductionModel
{
    private readonly int _nFft;
    private readonly int _winLength;
    private readonly int _hopLength;
    private readonly nn.Module<Tensor, Tensor> _encoder1;
    private readonly nn.Module<Tensor, Tensor> _encoder2;
    private readonly nn.Module<Tensor, Tensor> _encoder3;

    private readonly nn.Module<Tensor, Tensor> _decoder3;
    private readonly nn.Module<Tensor, Tensor> _decoder2;
    private readonly nn.Module<Tensor, Tensor> _decoder1;

    private readonly LSTM _blstm;
    private readonly Linear _projAfterLstm;
    private readonly Device _device;

    public DCCRNModel(Device? device = null, int nFft = 512, int inChannels = 2,
        int feat1 = 16, int feat2 = 32, int feat3 = 64, long lstmHidden = 128, int winLength = 512,
        int hopLength = 128) : base(nameof(DCCRNModel))
    {
        _nFft = nFft;
        _winLength = winLength;
        _hopLength = hopLength;
        _device = device ?? CPU;

        _encoder1 = nn.Sequential(
            ("conv", nn.Conv2d(inChannels, feat1, 3, 1, 1)),
            ("bn", nn.BatchNorm2d(feat1)),
            ("prelu", nn.PReLU(feat1, 0.25, _device))
        );

        _encoder2 = nn.Sequential(
            ("conv", nn.Conv2d(feat1, feat2, 3, 2, 1)), // kernel=3,stride=2,pad=1
            ("bn", nn.BatchNorm2d(feat2)),
            ("prelu", nn.PReLU(feat2, 0.25, _device))
        );

        _encoder3 = nn.Sequential(
            ("conv", nn.Conv2d(feat2, feat3, 3, 2, 1)), // kernel=3,stride=2,pad=1
            ("bn", nn.BatchNorm2d(feat3)),
            ("prelu", nn.PReLU(feat3, 0.25, _device))
        );

        int ConvOutSize(int inputSize, int kernel = 3, int stride = 1, int pad = 1, int dilation = 1)
        {
            // integer division in C# already floors for positives
            return (inputSize + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
        }
        
        int freqBins = _nFft / 2 + 1; // F = n_fft/2 + 1
        int f1 = ConvOutSize(freqBins, kernel: 3, stride: 1, pad: 1);
        int f2 = ConvOutSize(f1, kernel: 3, stride: 2, pad: 1);
        int f3 = ConvOutSize(f2, kernel: 3, stride: 2, pad: 1);

        long lstmInputSize = feat3 * Math.Max(1, f3);

        _blstm = nn.LSTM(lstmInputSize, lstmHidden, numLayers: 1, batchFirst: true, bidirectional: true);
        _projAfterLstm = nn.Linear((int)(lstmHidden * 2), (int)lstmInputSize);

        // === Decoder blocks ===
        _decoder3 = nn.Sequential(
            ("dconv", nn.ConvTranspose2d(feat3, feat2, 4, 2, 1)),
            ("bn", nn.BatchNorm2d(feat2)),
            ("prelu", nn.PReLU(feat2, 0.25, _device))
        );

        // После конкатенации по каналам мы ожидаем feat2 + feat2 = feat2*2 на вход decoder2
        _decoder2 = nn.Sequential(
            ("dconv", nn.ConvTranspose2d((feat2 * 2), feat1, 4, 2, 1)),
            ("bn", nn.BatchNorm2d(feat1)),
            ("prelu", nn.PReLU(feat1, 0.25, _device))
        );

        _decoder1 = nn.Sequential(
            ("conv", nn.Conv2d(feat1 * 2, 2, 1)),
            ("tanh", nn.Tanh())
        );
        
        RegisterComponents();
    }

    public string Name => name ?? nameof(DCCRNModel);

    /// <summary>
    /// Main inference function used by NoiseModelEvaluator.
    /// Takes mono waveform (float[]) and returns denoised waveform (float[]), same length.
    /// </summary>
    public float[] Process(float[] noisyWaveform, int sampleRate)
    {
        // 1) waveform -> stft (complex tensor)
        using var x = WaveformToComplexSpec(noisyWaveform); // shape [1,2,F,T]
        // move to device
        Tensor xDevice = x.to(_device);

        // 2) forward
        eval(); // inference mode
        using var withNoGrad = torch.no_grad(); // ensure no grad
        var estimatedSpec = forward(xDevice); // [1,2,F,T] complex representation (real/imag channels)

        // 3) move to CPU (if needed) and istft
        var estCpu = estimatedSpec.to(torch.CPU);

        float[] denoised = SpectrogramToWaveform(estCpu, noisyWaveform.Length);

        return denoised;
    }

    /// <summary>
    /// Load model weights saved as a state_dict (PyTorch compatible) or via TorchSharp save.
    /// Supports torch.load of a state_dict, and also supports TorchSharp load_py if available.
    /// </summary>
    public void LoadModel(string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");
        this.to(CPU);
        load(modelPath);
        Console.WriteLine($"Loaded TorchSharp .dat model from: {modelPath}");
    }

    /// <summary>
    /// Save model weights to path (state_dict). Compatible with PyTorch loading.
    /// </summary>
    public void SaveModel(string path)
    {
        this.to(CPU);
        save($"{path}/{Name}.dat");
    }

    /// <summary>
    /// For a sample run Process and write WAVs near original files.
    /// Uses AudioUtils.WriteMonoWav if available; otherwise tries a simple WAV writer fallback.
    /// </summary>
    public void ProcessAndWriteSample(IEnumerable<AudioData> sample)
    {
        foreach (var audioData in sample)
        {
            // Читаем моно сигнал
            float[] noisy = AudioUtils.ReadMonoWav(audioData.NoisyPath, out int sr);

            // Прогон через модель (Process возвращает float[] той же длины, что и вход)
            float[] denoised = Process(noisy, sr);

            // Обрезаем значения в допустимый диапазон [-1,1] во избежание клиппинга/артефактов при записи
            for (int i = 0; i < denoised.Length; i++)
            {
                float v = denoised[i];
                denoised[i] = v switch
                {
                    > 1f => 1f,
                    < -1f => -1f,
                    _ => denoised[i]
                };
            }

            // Подготовим путь для сохранения
            string inDir = Path.GetDirectoryName(audioData.NoisyPath) ?? ".";
            Directory.CreateDirectory(inDir); // на всякий случай

            string outName = $"{Path.GetFileNameWithoutExtension(audioData.FileName)}.{Name}.denoised.wav";
            string outPath = Path.Combine(inDir, outName);

            var waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(sr, 1);
            using (var writer = new WaveFileWriter(outPath, waveFormat))
            {
                // WriteSamples принимает float[] и записывает их как IEEE float
                writer.WriteSamples(denoised, 0, denoised.Length);
                writer.Flush();
            }

            Console.WriteLine($"[{Name}] Сохранён файл без шума: {outPath}");
        }
    }

    public override Tensor forward(Tensor input)
    {
        // assume input on correct device
        // 1) encoder
        var e1 = _encoder1.forward(input); // [B, feat1, F, T]
        var e2 = _encoder2.forward(e1); // [B, feat2, F/2, T]
        var e3 = _encoder3.forward(e2); // [B, feat3, F/4, T]

        // 2) prepare for LSTM: reshape to [B, T, feat3 * F_reduced]
        var shape = e3.shape; // [B, C3, F_reduced, T]
        long B = shape[0];
        long C3 = shape[1];
        long F_reduced = shape[2];
        long T = shape[3];

        // Permute to [B, T, C3, F_reduced] -> then flatten last two dims
        var x = e3.permute(0, 3, 1, 2); // [B, T, C3, F_reduced]
        var xFlatten = x.reshape(B, T, C3 * F_reduced); // [B, T, C3*F_reduced]

        // 3) BLSTM
        // NOTE: TorchSharp LSTM forward returns (output, (h, c)) — but direct tuple handling may need casting.
        // We'll call blstm.forward and try to extract first element.
        var blstmOut = _blstm.forward(xFlatten); // blstmOut имеет компилируемый тип ValueTuple<Tensor, (Tensor, Tensor)>
        Tensor blstmOutput = blstmOut.Item1; // берем первый элемент — output [B, T, hidden*2]

        // blstmOutput: [B, T, lstmHidden*2]
        // project back to C3 * F_reduced
        var proj = _projAfterLstm.forward(blstmOutput); // [B, T, C3*F_reduced]
        var projReshaped = proj.reshape(B, T, C3, F_reduced); // [B, T, C3, F_reduced]
        var bottleneck = projReshaped.permute(0, 2, 3, 1); // [B, C3, F_reduced, T]

        // 4) decoder with skip connections
        var d3 = _decoder3.forward(bottleneck); // upsample -> [B, feat2, F/2, T]

        // concat with e2 along channel dim (1)
        // ensure same spatial size
        long diffF2 = e2.shape[2] - d3.shape[2];
        long diffT2 = e2.shape[3] - d3.shape[3];
        if (diffF2 != 0 || diffT2 != 0)
        {
            // pad right and bottom if needed
            d3 = nn.functional.pad(d3, new long[] { 0, diffT2, 0, diffF2 });
        }

        var cat2 = cat(new Tensor[] { d3, e2 }, 1); // [B, feat2*2, F/2, T]
        var d2 = _decoder2.forward(cat2); // -> [B, feat1, F, T]

        // concat with e1 along channel dim (1)
        // ensure same spatial size
        long diffF1 = e1.shape[2] - d2.shape[2];
        long diffT1 = e1.shape[3] - d2.shape[3];
        if (diffF1 != 0 || diffT1 != 0)
        {
            // pad right and bottom if needed
            d2 = nn.functional.pad(d2, new long[] { 0, diffT1, 0, diffF1 });
        }

        var cat1 = cat(new Tensor[] { d2, e1 }, 1); // [B, feat1*2, F, T]
        var outMask = _decoder1.forward(cat1);   // [B, 2, F, T] (tanh) -> CRM mask (real, imag)


        // 5) Multiply noisy complex spectrum by mask: input complex is [B,2,F,T]
        // complex multiplication: (a+jb)*(c+jd) = (ac - bd) + j(ad + bc)

        // Split input into real and imag: each of shape [B,1,F,T]
        var noisySplits = input.split(1, dim: 1);
        var noisyR = noisySplits[0].squeeze(1); // [B, F, T]
        var noisyI = noisySplits[1].squeeze(1); // [B, F, T]

        // Split mask into real and imag channels: each [B,1,F,T]
        var maskSplits = outMask.split(1, dim: 1);
        var maskR = maskSplits[0].squeeze(1); // [B, F, T]
        var maskI = maskSplits[1].squeeze(1); // [B, F, T]

        // compute complex multiplication (element-wise)
        var estR = noisyR.mul(maskR).sub(noisyI.mul(maskI)); // (ac - bd)
        var estI = noisyR.mul(maskI).add(noisyI.mul(maskR)); // (ad + bc)

        // restore channel dim -> [B,1,F,T]
        var estRExp = estR.unsqueeze(1);
        var estIExp = estI.unsqueeze(1);

        // concat to [B,2,F,T]
        return cat([estRExp, estIExp], dim: 1);
    }

    /// <summary>
    /// Convert waveform float[] (mono) -> Tensor complex spectrogram shaped [1,2,F,T] (real/imag).
    /// </summary>
    private Tensor WaveformToComplexSpec(float[] waveform)
    {
        // normalize to float tensor
        Tensor floatTensor = tensor(waveform, dtype: ScalarType.Float32, device: CPU);
        // add batch dim
        Tensor batched = floatTensor.unsqueeze(0); // [1, N]

        // create Hann window on CPU (then move to device when used in stft)
        Tensor window = hann_window(_winLength, dtype: ScalarType.Float32);

        Tensor spec = stft(batched, _nFft, _hopLength, _winLength, window: window, center: true, return_complex: false);
        // [1, F, T, 2]
        // reorder to [B, 2, F, T]
        Tensor specPerm = spec.permute(0, 3, 1, 2); // [B, 2, F, T]
        return specPerm;
    }

    /// <summary>
    /// Convert estimated complex spec Tensor [1,2,F,T] -> waveform float[] of required length.
    /// </summary>
    public float[] SpectrogramToWaveform(Tensor complexSpec, int desiredLength)
    {
        // complexSpec: [B,2,F,T] on some device
        Tensor specPerm = complexSpec.permute(0, 2, 3, 1).contiguous(); // [B, F, T, 2]
        Tensor complexTensor = view_as_complex(specPerm);

        // Hann window on same device as complex tensor
        Tensor window = hann_window(_winLength, dtype: ScalarType.Float32, device: complexTensor.device);

        Tensor waveform = istft(
            complexTensor,
            _nFft,
            _hopLength,
            _winLength,
            window: window,
            center: true,
            normalized: false,
            length: desiredLength,
            return_complex: false
        );

        return waveform.to(CPU).data<float>().ToArray();
    }
}
