namespace Glacier.Clavier.Demo;

using System;
using System.Diagnostics;
using Glacier.Clavier.Engine;
using Glacier.Clavier.Model;
using Glacier.Gpu.Drivers;
using Glacier.Gpu.Engines;

class Program
{
    static void Main(string[] args)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("================================================================================");
        Console.WriteLine("         GLACIER.CLAVIER: SYSTEM-1 CALIBRATED POLICY ENGINE DEMO                ");
        Console.WriteLine("    Non-Autoregressive Single Forward Pass (< 2 ms Turnaround on RTX 3060)      ");
        Console.WriteLine("================================================================================\n");
        Console.ResetColor();

        Console.WriteLine($"[ENVIRONMENT] OS: {Environment.OSVersion}, .NET: {Environment.Version}, Cores: {Environment.ProcessorCount}\n");

        // 0. Hardware Acceleration & Unified Memory Evaluation (AMD 890M / ROCm / HIP)
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine("[0/3] Probing Hardware Accelerators & Unified Memory Architecture...");
        Console.ResetColor();

        try
        {
            if (HipDriver.IsAvailable())
            {
                using var amd = new AmdRdnaEngine();
                amd.Initialize();
                var dev = amd.DeviceInfo;
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"  -> AMD Accelerator Detected: {dev.DeviceName}");
                Console.WriteLine($"  -> Microarchitecture:        {dev.Architecture}");
                Console.WriteLine($"  -> Device Type:              {dev.DeviceType} (Unified Memory: {dev.IsUnifiedMemory})");
                Console.WriteLine($"  -> Compute Units:            {dev.ComputeUnitsOrSms} CUs");
                Console.WriteLine($"  -> Addressable Host VRAM:    {dev.TotalMemoryBytes / (1024 * 1024 * 1024.0):F2} GB");

                Console.WriteLine("  -> Testing Zero-Copy Coherency & Host/GPU Direct Sharing...");
                var (sharingLatencyUs, verified) = amd.TestZeroCopyDirectAccess();
                Console.WriteLine($"     Zero-Copy Direct Coherency: {(verified ? "PASS" : "FAIL")}");
                Console.WriteLine($"     Direct Host/GPU Access Latency: {sharingLatencyUs:F3} µs (0 ns PCIe Staging Penalty)");

                Console.WriteLine("  -> Benchmarking Sustained Unified Memory Bandwidth (256 MB buffer)...");
                var (bandwidthGbps, avgMs) = amd.BenchmarkUnifiedBandwidth(256);
                Console.WriteLine($"     Sustained Bandwidth:       {bandwidthGbps:F2} GB/s");
                Console.WriteLine($"     256 MB Turnaround:         {avgMs:F2} ms");
                Console.ResetColor();
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.DarkYellow;
                Console.WriteLine("  -> AMD HIP Native Driver not active in this execution path; running host SIMD mode.");
                Console.ResetColor();
            }
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.DarkYellow;
            Console.WriteLine($"  -> Hardware acceleration note: {ex.Message}");
            Console.ResetColor();
        }
        Console.WriteLine();

        // 1. Initialize Clavier Policy Engine
        using var session = new ClavierSession(
            embeddingDim: 768,
            hiddenDim: 256,
            numActions: 8,
            temperature: 1.1f);

        string[] candidateActions =
        [
            "ENGAGE_TARGET",
            "TACTICAL_RETREAT",
            "TAKE_COVER",
            "RELOAD_WEAPON",
            "DEPLOY_SHIELD",
            "SCAN_PERIMETER",
            "HOLD_POSITION",
            "CALL_REINFORCEMENTS"
        ];

        string[] simulationStates =
        [
            "Enemy squad approaching from West flank at 25m. Armor status: 92%, Primary Ammo: 100%. Distance to cover: 3m.",
            "Heavy artillery strike inbound at current coordinates. Shields down to 18%. Mobility impaired.",
            "Target spotted in open terrain without cover. Distance: 40m. Line of sight clear. Full magazine loaded.",
            "Weapon magazine empty (0/30). Hostile unit advancing aggressively at 10m. Sidearm ready.",
            "Radar telemetry clear. Zero hostiles in 100m radius. Ambient visibility degrading due to fog.",
            "Surrounded by multiple adversaries. Health critical (12%). Extraction beacon operational."
        ];

        Console.WriteLine($"[1/3] Simulating Jev Decision Primitives (Choice, Noul, Score) across {simulationStates.Length} Scenarios...\n");

        string hypothesis = "Immediate catastrophic threat requiring urgent intervention";
        string criteria = "Tactical Hazard & Urgency Index";

        for (int i = 0; i < simulationStates.Length; i++)
        {
            var state = simulationStates[i].AsSpan();

            // 1. Primitive 1: Choice (Discrete Action Selection)
            var choice = session.DecideWithDistribution(state, candidateActions);

            // 2. Primitive 2: Noul (Binary Hypothesis Verification)
            var noul = session.VerifyWithDetails(state, hypothesis.AsSpan(), threshold: 0.5f);

            // 3. Primitive 3: Score (Continuous Scale Evaluation)
            var score = session.ScoreWithDetails(state, criteria.AsSpan(), min: 0.0f, max: 100.0f);

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"[State {i + 1}]: \"{simulationStates[i]}\"");
            
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"  [Choice] Selected Action:  [{choice.ActionName}] (ActionId: {choice.Decision.ActionId}, Conf: {choice.Decision.Confidence * 100f:F1}%, {choice.LatencyMs:F3} ms)");
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"  [Noul]   Threat Imminent:  {(noul.IsAffirmative ? "YES" : "NO")} (Prob: {noul.Probability * 100f:F1}%, Certainty: {noul.Confidence * 100f:F1}%, {noul.LatencyMs:F3} ms)");
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine($"  [Score]  Hazard Severity:  {score.Value:F1}/100.0 (Confidence: {score.Confidence * 100f:F1}%, {score.LatencyMs:F3} ms)");

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine("  -> Choice Action Distribution:");
            for (int a = 0; a < candidateActions.Length; a++)
            {
                int barLen = (int)(choice.Probabilities[a] * 30);
                string bar = new string('█', barLen);
                Console.WriteLine($"     {candidateActions[a],-20} | {choice.Probabilities[a] * 100f,5:F1}% {bar}");
            }
            Console.WriteLine();
        }

        // 2. High-Frequency Benchmark across all 3 primitives (3,000 inferences)
        Console.ResetColor();
        Console.WriteLine("[2/3] Running High-Frequency Benchmark (1,000 Choice + 1,000 Noul + 1,000 Score)...");

        // Warmup JIT
        for (int i = 0; i < 20; i++)
        {
            _ = session.Decide(simulationStates[0].AsSpan(), candidateActions);
            _ = session.Verify(simulationStates[0].AsSpan(), hypothesis.AsSpan());
            _ = session.Score(simulationStates[0].AsSpan(), criteria.AsSpan(), 0f, 100f);
        }

        const int iterations = 1000;
        
        // Choice Benchmark
        var swChoice = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            _ = session.Decide(simulationStates[i % simulationStates.Length].AsSpan(), candidateActions);
        }
        swChoice.Stop();

        // Noul Benchmark
        var swNoul = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            _ = session.Verify(simulationStates[i % simulationStates.Length].AsSpan(), hypothesis.AsSpan());
        }
        swNoul.Stop();

        // Score Benchmark
        var swScore = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            _ = session.Score(simulationStates[i % simulationStates.Length].AsSpan(), criteria.AsSpan(), 0f, 100f);
        }
        swScore.Stop();

        double choiceLatencyUs = (swChoice.Elapsed.TotalMilliseconds / iterations) * 1000.0;
        double choiceQps = iterations / swChoice.Elapsed.TotalSeconds;

        double noulLatencyUs = (swNoul.Elapsed.TotalMilliseconds / iterations) * 1000.0;
        double noulQps = iterations / swNoul.Elapsed.TotalSeconds;

        double scoreLatencyUs = (swScore.Elapsed.TotalMilliseconds / iterations) * 1000.0;
        double scoreQps = iterations / swScore.Elapsed.TotalSeconds;

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"\n[BENCHMARK RESULTS (3,000 Inferences)]");
        Console.WriteLine($"  [Choice] Latency: {choiceLatencyUs:F1} µs ({choiceLatencyUs / 1000.0:F3} ms) | Throughput: {choiceQps:F0} decisions/sec (8 bytes payload)");
        Console.WriteLine($"  [Noul]   Latency: {noulLatencyUs:F1} µs ({noulLatencyUs / 1000.0:F3} ms) | Throughput: {noulQps:F0} verifications/sec (8 bytes payload)");
        Console.WriteLine($"  [Score]  Latency: {scoreLatencyUs:F1} µs ({scoreLatencyUs / 1000.0:F3} ms) | Throughput: {scoreQps:F0} ratings/sec (8 bytes payload)");
        Console.ResetColor();
        Console.ResetColor();
    }
}
