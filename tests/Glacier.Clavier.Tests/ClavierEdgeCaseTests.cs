namespace Glacier.Clavier.Tests;

using System;
using System.Threading.Tasks;
using Glacier.Clavier.Engine;
using Glacier.Clavier.Model;
using Xunit;

public class ClavierEdgeCaseTests
{
    [Fact]
    public void ClavierSession_Constructor_ValidatesArguments()
    {
        Assert.Throws<ArgumentException>(() => new ClavierSession(embeddingDim: 0, hiddenDim: 128, numActions: 4));
        Assert.Throws<ArgumentException>(() => new ClavierSession(embeddingDim: 128, hiddenDim: 0, numActions: 4));
        Assert.Throws<ArgumentException>(() => new ClavierSession(embeddingDim: 128, hiddenDim: 128, numActions: 0));

        // When temperature <= 0, head defaults to 1.0f
        var head = new ClavierDecisionHead(128, 64, 4, temperature: -1.0f);
        Assert.Equal(1.0f, head.Temperature);

        // Property setter throws on non-positive temperature
        Assert.Throws<ArgumentOutOfRangeException>(() => head.Temperature = 0.0f);
    }

    [Fact]
    public void ClavierSession_Decide_EmptyActions_FallsBackToConfiguredNumActions()
    {
        using var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 4);
        string[] emptyActions = [];

        var decision = session.Decide("state".AsSpan(), emptyActions);
        Assert.True(decision.ActionId < 4);
        Assert.InRange(decision.Confidence, 0.0f, 1.0f);
    }

    [Fact]
    public void ClavierSession_Decide_EmptyState_ExecutesSuccessfully()
    {
        using var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 4);
        string[] actions = ["ACT_A", "ACT_B"];

        var decision = session.Decide(ReadOnlySpan<char>.Empty, actions);
        Assert.True(decision.ActionId < (uint)actions.Length);
        Assert.InRange(decision.Confidence, 0.0f, 1.0f);
    }

    [Fact]
    public void ClavierSession_Disposal_IsIdempotent()
    {
        var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 4);
        session.Dispose();
        session.Dispose(); // second call should not throw
    }

    [Fact]
    public void ClavierSession_AfterDispose_ThrowsObjectDisposedException()
    {
        var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 4);
        session.Dispose();

        string[] actions = ["A", "B"];
        Assert.Throws<ObjectDisposedException>(() => session.Decide("state".AsSpan(), actions));
    }

    [Fact]
    public void ClavierSession_ConcurrentDecide_IsThreadSafe()
    {
        using var session = new ClavierSession(embeddingDim: 128, hiddenDim: 64, numActions: 8);
        string[] actions = ["N", "S", "E", "W"];

        Parallel.For(0, 50, i =>
        {
            string state = $"Agent #{i} at coordinate ({i * 2}, {i * 3})";
            var dec = session.Decide(state.AsSpan(), actions);
            Assert.True(dec.ActionId < (uint)actions.Length);
            Assert.InRange(dec.Confidence, 0.0f, 1.0f);
        });
    }
}
